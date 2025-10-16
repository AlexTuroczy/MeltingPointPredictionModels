import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import logging
import torch
from sklearn.model_selection import train_test_split
from src.train_utils import form_graph_data, cross_validation, train, predict_test_set, set_seed
import yaml


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

def main():
    
    # read in configurable parameters
    with open("experiments/GCN_Experiment_1/config.yaml", "r") as f:
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
                     lr=0.01, weight_decay=1e-4, epochs=200, batch_size=16, device='cpu', seed=seed)
    logging.info(f"Average validation loss: {avg_val}")
    train_data, val_data= train_test_split(graph_list, train_size=0.95, random_state=seed, shuffle=True)

    model = GCNRegressor(**model_parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-8)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, generator=generator)
    train(model, train_loader, optimizer, criterion, val_loader=val_loader, scheduler=scheduler, epochs=200)

    predict_test_set(model, path_to_test, output_path, device="cpu", to_kaggle=True)    

if __name__ == "__main__":
    main()