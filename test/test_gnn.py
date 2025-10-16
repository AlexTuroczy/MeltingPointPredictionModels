from src.train_utils import smiles_to_torch_geometric_data
import torch
from collections import defaultdict

def test_smiles_to_graph():

    # obtain initial symbol embedding
    symbol_to_one_hot = defaultdict(int)
    symbol_to_one_hot["C"] = torch.tensor(1, dtype=torch.long)
    formula = "CCCF"
    data = smiles_to_torch_geometric_data(formula, symbol_to_one_hot)

    # shape tests
    assert(data.x.shape == torch.Size([4])) # 4 nodes, initially 1 feature
    assert(data.edge_index.shape == torch.Size([2, 6])) # 4 arcs, each having 2 end points

    # value tests
    print(data.x)
    assert(torch.equal(data.x, torch.tensor([1, 1, 1, 0], dtype=torch.long)))
