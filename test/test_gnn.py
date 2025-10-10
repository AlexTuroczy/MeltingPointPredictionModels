from src.gnn import smiles_to_torch_geometric_data
import torch

def test_smiles_to_graph():

    # obtain initial symbol embedding
    symbol_to_one_hot = {"C": torch.tensor([1, 0], dtype=torch.long)}
    default_value = torch.tensor([0, 1], dtype=torch.long)

    formula = "CCCF"
    data = smiles_to_torch_geometric_data(formula, symbol_to_one_hot, default_value)

    # shape tests
    assert(data.x.shape == torch.Size([4, 2])) # 4 nodes, initially 2 features
    assert(data.edge_index.shape == torch.Size([2, 6])) # 4 arcs, each having 2 end points

    # value tests
    print(data.x)
    assert(torch.equal(data.x, torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=torch.long)))
