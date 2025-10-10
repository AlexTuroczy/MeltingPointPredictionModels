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
from collections import defaultdict
import torch.nn.functional as F

# --------------------- MODELS ---------------------
class GCNRegressor(nn.Module):
    def __init__(self, num_embeddings, embed_dim, hidden_channels, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embed_dim, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.fc = nn.Linear(hidden_channels, 1)
        self.reset_parameters()

    def forward(self, x, edge_index, batch):
        # Map integer atom IDs -> embedding vectors
        x = self.embedding(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()

# --------------------- DATA WRANGLING ---------------------
def symbols_map(data_df: pd.DataFrame):
    symbols = set()
    for smiles in data_df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        symbols.update({atom.GetSymbol() for atom in mol.GetAtoms()})
    symbols_to_int_map = defaultdict(int)
    for idx, val in enumerate(symbols):
        symbols_to_int_map[val] = idx + 1  # reserve 0 for padding if needed
    return symbols_to_int_map

def smiles_to_torch_geometric_data(formula: str, symbols_to_int_map: dict):
    mol = Chem.MolFromSmiles(formula)
    atoms, bonds = mol.GetAtoms(), mol.GetBonds()

    x = torch.tensor(
        [symbols_to_int_map[atom.GetSymbol()] for atom in atoms],
        dtype=torch.long
    )

    edge_index = torch.tensor(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds] +
        [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] for bond in bonds],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

def form_graph_data(data_path: str):
    data_df = pd.read_csv(data_path)
    symbols_to_int_map = symbols_map(data_df)

    graph_list = []
    for idx, smiles in enumerate(data_df["SMILES"]):
        data = smiles_to_torch_geometric_data(smiles, symbols_to_int_map)
        if "Tm" in data_df.columns:
            data.y = torch.tensor([data_df["Tm"][idx]], dtype=torch.float)
        graph_list.append(data)

    return graph_list

# --------------------- TRAIN / EVALUATE ---------------------
def train(model, data_loader, optimizer, criterion, epochs=100, device='cpu'):
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
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(data_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return model

def evaluate(model, data_loader, criterion, device='cpu', mean=0.0, std=1.0, denormalize=False):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    preds, targets = [], []

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

            loss = criterion(y_pred, y_true)
            total_loss += loss.item() * batch.num_graphs
            preds.append(y_pred.cpu())
            targets.append(y_true.cpu())

    avg_loss = total_loss / len(data_loader.dataset)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return avg_loss, preds, targets

# --------------------- CROSS-VALIDATION ---------------------
def cross_validation(model, graph_list, optimizer_class, criterion, k_folds,
                     lr=0.01, weight_decay=1e-4, epochs=100, batch_size=16, device='cpu'):
    # Store original targets
    for g in graph_list:
        g.y_orig = g.y.clone()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    avg_loss = 0.0
    fold = 0

    for train_idx, val_idx in kf.split(graph_list):
        fold += 1
        print(f"Fold {fold}")

        train_graphs = [graph_list[i] for i in train_idx]
        val_graphs   = [graph_list[i] for i in val_idx]

        # Restore original targets
        for g in train_graphs: g.y = g.y_orig.clone()
        for g in val_graphs: g.y = g.y_orig.clone()

        # Compute mean/std on training fold
        y_train = torch.stack([g.y for g in train_graphs])
        mean, std = y_train.mean(), y_train.std()
        print(mean, std)

        # Normalize targets
        for g in train_graphs: g.y = (g.y - mean) / std
        for g in val_graphs: g.y = (g.y - mean) / std

        # DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

        # Reset model + optimizer per fold
        model.reset_parameters()
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        model = train(model, train_loader, optimizer, criterion, epochs=epochs, device=device)
        loss, _, _ = evaluate(model, val_loader, criterion, device=device, mean=mean, std=std, denormalize=False)
        avg_loss += loss
        print(f"Fold {fold} Validation Loss (denormalized): {loss:.4f}\n")

    avg_loss /= k_folds
    return avg_loss

# --------------------- MAIN ---------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_data_path = "data/melting-point/train.csv"
    graph_list = form_graph_data(train_data_path)

    model = GCNRegressor(num_embeddings=11, embed_dim=8, hidden_channels=8, num_layers=4)
    criterion = nn.MSELoss()

    k_folds = 5
    cross_validation(model, graph_list, torch.optim.Adam, criterion, k_folds,
                     epochs=100, lr=1e-4, weight_decay=1e-6, batch_size=16)

if __name__ == "__main__":
    main()