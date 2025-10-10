import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from rdkit import Chem
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
import torch.nn.functional as F

# --------------------- MODELS ---------------------
class GCNRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)  # regression output

        # Automatically initialize parameters
        self.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out.squeeze()

    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()
# --------------------- DATA WRANGLING ---------------------


def symbols_to_one_hot_map(data_df: pd.DataFrame) -> tuple[dict, torch.tensor]:
    symbols = set()
    for smiles in data_df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        symbols.update({atom.GetSymbol() for atom in mol.GetAtoms()})
    num_elements = len(symbols)
    symbol_to_one_hot = {val: F.one_hot(torch.tensor(idx), num_elements+1).float() for idx, val in enumerate(symbols)}
    default_value = F.one_hot(torch.tensor(num_elements), num_elements+1).float()
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
    for idx, smiles in enumerate(data_df["SMILES"]):
        data = smiles_to_torch_geometric_data(smiles, symbol_to_one_hot, default_value)

        # add response variable if in the training set
        if "Tm" in data_df.columns:
            data.y = torch.tensor(data_df["Tm"][idx], dtype=torch.float)
        graph_list.append(data)

    return graph_list

# --------------------- MODEL TRAINING ---------------------

def train(model, data_loader, optimizer, criterion, epochs=100, device='cpu'):
    """
    Simple training loop for a PyTorch Geometric model.

    Args:
        model: PyTorch model (e.g., GCNRegressor)
        data_loader: PyG DataLoader yielding batches of Data objects
        optimizer: PyTorch optimizer
        criterion: loss function (e.g., nn.MSELoss for regression)
        epochs: number of training epochs
        device: 'cpu' or 'cuda'

    Returns:
        model: trained model
    """
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs  # scale by number of graphs

        total_loss /= len(data_loader.dataset)  # average over dataset

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

def evaluate(model, data_loader, criterion, device='cpu', mean=0.0, std=1.0, denormalize=False):
    """
    Evaluate a PyTorch Geometric model on a dataset.

    Args:
        model: PyTorch model
        data_loader: PyG DataLoader yielding batches of Data objects
        criterion: loss function (e.g., nn.MSELoss for regression)
        device: 'cpu' or 'cuda'
        mean: training fold mean for target normalization
        std: training fold std for target normalization
        denormalize: if True, outputs and targets are rescaled to original range

    Returns:
        avg_loss: average loss over the dataset
        preds: tensor of predictions
        targets: tensor of ground truth values
    """
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

            if denormalize:
                y_pred = out * std + mean
                y_true = batch.y * std + mean
            else:
                y_pred = out
                y_true = batch.y

            # compute loss in the correct scale
            loss = criterion(y_pred, y_true)
            total_loss += loss.item() * batch.num_graphs  # scale by number of graphs

            # accumulate predictions and targets
            preds.append(y_pred.cpu())
            targets.append(y_true.cpu())

    avg_loss = total_loss / len(data_loader.dataset)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    return avg_loss, preds, targets


def cross_validation(model, graph_list, optimizer_class, criterion, k_folds,
                     lr=0.01, epochs=100, batch_size=16, device='cpu'):
    """
    Perform K-fold cross-validation with per-fold target normalization.

    Args:
        model: PyTorch model
        graph_list: list of PyG Data objects
        optimizer_class: optimizer class (e.g., torch.optim.Adam)
        criterion: loss function
        k_folds: number of folds
        lr: learning rate
        epochs: training epochs per fold
        batch_size: DataLoader batch size
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss: mean validation loss across folds
    """
    # Store original targets to avoid contamination across folds
    for g in graph_list:
        g.y_orig = g.y.clone()

    assert k_folds > 0
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    avg_loss = 0.0
    fold = 0

    for train_idx, val_idx in kf.split(graph_list):
        fold += 1
        print(f"Fold {fold}")

        # Split graphs
        train_graphs = [graph_list[i] for i in train_idx]
        val_graphs   = [graph_list[i] for i in val_idx]

        # Restore original targets before normalization
        for g in train_graphs:
            g.y = g.y_orig.clone()
        for g in val_graphs:
            g.y = g.y_orig.clone()

        # Compute mean/std from training fold only
        y_train = torch.stack([g.y for g in train_graphs])
        mean, std = y_train.mean(), y_train.std()

        # Normalize targets per fold
        for g in train_graphs:
            g.y = (g.y - mean) / std
        for g in val_graphs:
            g.y = (g.y - mean) / std

        # Create DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

        # Initialize model and optimizer for this fold
        model.reset_parameters()
        optimizer = optimizer_class(model.parameters(), lr=lr)

        # Train fold
        model = train(model, train_loader, optimizer, criterion, epochs=epochs, device=device)

        # Evaluate fold (denormalized)
        loss, _, _ = evaluate(model, val_loader, criterion,
                              device=device, mean=mean, std=std, denormalize=True)
        avg_loss += loss
        print(f"Fold {fold} Validation Loss (denormalized): {loss:.4f}\n")

    avg_loss /= k_folds
    return avg_loss

def main():
    logging.basicConfig(level=logging.INFO, \
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # obtain training_data
    train_data_path = Path("data/melting-point/train.csv")
    graph_list = form_graph_data(train_data_path)

    model = GCNRegressor(in_channels=11, hidden_channels=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    k_folds = 5
    cross_validation(model, graph_list, torch.optim.Adam, criterion, k_folds)

if __name__ == "__main__":
    main()