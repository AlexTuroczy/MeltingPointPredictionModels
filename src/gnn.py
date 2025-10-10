import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from rdkit import Chem
from torch_geometric.data import Data
import torch.nn.functional as F

def symbols_to_one_hot_map(data_df: pd.DataFrame) -> tuple[dict, torch.tensor]:
    symbols = set()
    for smiles in data_df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        symbols.update({atom.GetSymbol() for atom in mol.GetAtoms()})
    num_elements = len(symbols)
    symbol_to_one_hot = {val: F.one_hot(torch.tensor(idx), num_elements+1) for idx, val in enumerate(symbols)}
    default_value = F.one_hot(torch.tensor(num_elements), num_elements+1)
    return symbol_to_one_hot, default_value


def smiles_to_torch_geometric_data(formula: str, symbol_to_one_hot: dict, default_value: torch.tensor):
    mol = Chem.MolFromSmiles(formula)
    atoms, bonds = mol.GetAtoms(), mol.GetBonds()

    x = torch.stack(
    [symbol_to_one_hot.get(atom.GetSymbol(), default_value) for atom in atoms],
    )
    # Node feature matrix with shape [num_nodes, num_node_features]

    # Create edge index
    edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds] +
                            [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] for bond in bonds], dtype=torch.long).t().contiguous()
    # Graph connectivity in COO format with shape [2, num_edges]

    # Graph object
    data = Data(x=x, edge_index=edge_index)
    return data


def form_graph_data(data_path: pd.DataFrame) -> list[Data]:

    try:
        data_df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logging.error(f"Path not found, ensure path set correctly: {e}")
        raise

    # create mapping from element symbols to one-hot encoding
    symbol_to_one_hot, default_value = symbols_to_one_hot_map(data_df)

    graph_list = []
    for smiles in data_df["SMILES"]:
        data = smiles_to_torch_geometric_data(smiles, symbol_to_one_hot, default_value)
        graph_list.append(data)


    return graph_list
    
    

def train(model, data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[data.validation_mask], data.y[data.validation_mask])
    return loss

def cross_validation(model, data, optimizer, criterion, k_folds):

    assert(k_folds > 0)

    avg_loss = 0.0
    for i in range(k_folds):

        """
        To do: set train_mask and validation_mask
        """

        model.reset_parameters()
        model = train(model, data, optimizer, criterion, epochs=100)
        avg_loss += evaluate(model, data, criterion)

    avg_loss /= k_folds
    return avg_loss

def main():
    logging.basicConfig(level=logging.INFO, \
                        format='%(asctime)s - %(levelname)s - %(message)s')

    train_data_path = Path("data/melting-point/train.csv")
    form_graph_data(train_data_path)



if __name__ == "__main__":
    main()