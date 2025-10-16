import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import logging
import torch
from sklearn.model_selection import train_test_split
from src.train_utils import form_graph_data, cross_validation, train, predict_test_set, set_seed
import yaml
from pathlib import Path


class GCNRegressor(nn.Module):
    def __init__(self, num_embeddings, embed_dim, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embed_dim, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def forward(self, x, edge_index, batch):
        # Map integer atom IDs -> embedding vectors
        x = self.embedding(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        #x = self.dropout(x)
        return self.fc(x).squeeze(-1)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()

def main():
    
    # read in configurable parameters
    with open(Path(__file__).parent / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_data_path = config["train_data_path"]
    path_to_test = config["path_to_test"]
    output_path = config["output_path"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    seed = config["seed"]
    model_parameters = config["model_parameters"]
    logging_directory = config["logging_directory"]
    to_kaggle = config["to_kaggle"]
    epochs = config["epochs"]
    kaggle_message = config["kaggle_message"]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
        logging.FileHandler(logging_directory),  # write logs to this file
        logging.StreamHandler()                     # still print to console
    ])

    graph_list = form_graph_data(train_data_path)
    criterion = nn.MSELoss()
    
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    


    avg_val = cross_validation(GCNRegressor, model_parameters, graph_list, torch.optim.Adam, criterion, 5,
                     lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=16, device='cpu', seed=seed)
    logging.info(f"Average validation loss: {avg_val}")
    


    train_data, val_data= train_test_split(graph_list, train_size=0.95, random_state=seed, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, generator=generator)

    predict_test_set(GCNRegressor, model_parameters, train_loader, val_loader, criterion, path_to_test, output_path, seeds = [1,2,3,4,5], device="cpu", to_kaggle=to_kaggle, kaggle_message=kaggle_message, epochs=epochs, lr=lr, weight_decay=weight_decay)    

if __name__ == "__main__":
    main()