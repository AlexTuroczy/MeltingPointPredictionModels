"""
train_utils.py

Contains helper functions for training, evaluating, and performing cross-validation
for graph neural network models on molecular datasets.

Functions include:
- set_seed: Make experiments deterministic.
- form_graph_data / smiles_to_torch_geometric_data: Convert SMILES strings to PyTorch Geometric graphs.
- train: Standard training loop with optional validation and scheduler support.
- evaluate: Evaluate model performance on a dataset.
- cross_validation: K-fold cross-validation with per-fold target normalization.
- predict_test_set: Make predictions on a test set and save results.

Author: Alexander Turoczy
Date: 2025-10-16
"""

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from collections import defaultdict
import os
import subprocess
from pathlib import Path
import logging

def set_seed(seed=0):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# --------------------- DATA WRANGLING ---------------------

def symbols_map(data_df: pd.DataFrame):
    symbols = set()
    for smiles in data_df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        symbols.update({atom.GetSymbol() for atom in mol.GetAtoms()})
    symbols_to_int_map = defaultdict(int)
    symbols = sorted(symbols)
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
def train(model, train_loader, optimizer, criterion, val_loader=None, scheduler=None, epochs=100, device="cpu"):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)

        # optional validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    total_val_loss += loss.item() * batch.num_graphs
            val_loss = total_val_loss / len(val_loader.dataset)

        # Step scheduler correctly
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

        if epoch % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                + (f" - Val Loss: {val_loss:.4f}" if val_loss is not None else "")
                + f" - LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    return model

def evaluate(model, data_loader, criterion, device='cpu'):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

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
def cross_validation(model_class, model_parameters, graph_list, optimizer_class, criterion, k_folds,
                     lr=0.01, weight_decay=1e-4, epochs=100, batch_size=16, device='cpu', seed=0):
    
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Store original targets
    for g in graph_list:
        g.y_orig = g.y.clone()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    avg_loss = 0.0
    fold = 0

    for train_idx, val_idx in kf.split(graph_list):
        fold += 1
        logging.info(f"Fold {fold}")

        train_graphs = [graph_list[i] for i in train_idx]
        val_graphs   = [graph_list[i] for i in val_idx]

        # Restore original targets
        for g in train_graphs: g.y = g.y_orig.clone()
        for g in val_graphs: g.y = g.y_orig.clone()

        # DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader   = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, generator=generator)

        # Reset model + optimizer per fold
        model = model_class(**model_parameters)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-8)

        model = train(model, train_loader, optimizer, criterion, val_loader=val_loader, scheduler=scheduler, epochs=epochs)
        loss, _, _ = evaluate(model, val_loader, criterion, device=device)
        avg_loss += loss
        print(f"Fold {fold} Validation Loss: {loss:.4f}\n")

    avg_loss /= k_folds
    return avg_loss

def predict_test_set(model_class, model_parameters, train_loader, val_loader, criterion, path_to_test, output_path, seeds = [0], device="cpu", to_kaggle=False, kaggle_message=None, weight_decay=1e-4, epochs=200, lr=1e-3): 

    models = []
    for i, seed in enumerate(seeds):
        logging.info(f"Train model {i+1} of {len(seeds)}")
        torch.manual_seed(seed)
        model = model_class(**model_parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        train(model, train_loader, optimizer, criterion, val_loader=val_loader, scheduler=scheduler, epochs=epochs)
        models.append(model)
    
    
    # Load test data
    test_df = pd.read_csv(path_to_test)
    graph_list = form_graph_data(path_to_test)

    # DataLoader
    test_loader = DataLoader(graph_list, batch_size=16, shuffle=False)

    # Predict
    
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outs = []
            for model in models:
                model.eval()
                out = model(batch.x, batch.edge_index, batch.batch)
                outs.append(out)
            preds.append(torch.stack(outs, dim=0).mean(0).cpu())

    preds = torch.cat(preds).numpy()

    # Build submission DataFrame
    submission = pd.DataFrame({
        "id": test_df["id"],
        "Tm": preds
    })

    # Ensure path exists
    os.makedirs(Path("out"), exist_ok=True)

    # Save
    submission.to_csv(output_path, index=False)

    if to_kaggle and (kaggle_message is not None):
        # submit to kaggle
        competition_slug = "melting-point"

        subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", competition_slug,
        "-f", output_path,
        "-m", kaggle_message])