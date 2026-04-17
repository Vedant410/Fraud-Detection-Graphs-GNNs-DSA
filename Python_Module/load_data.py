import pandas as pd
import torch
from torch_geometric.data import Data

def load_data():
    edges = pd.read_csv("../data/edges.csv")
    features = pd.read_csv("../data/features.csv")
    labels = pd.read_csv("../data/labels.csv")

    num_nodes = len(features)

    # Node features: all columns except 'node'
    x = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)

    # Edge index
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # Labels: not all nodes have labels (78,499 nodes but only 50,000 labeled)
    # Create a label tensor for ALL nodes, default to 0
    y = torch.zeros(num_nodes, 1, dtype=torch.float)

    # Create a mask to track which nodes have labels (for training/evaluation)
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Fill in the labeled nodes
    for _, row in labels.iterrows():
        node_id = int(row["node"])
        if node_id < num_nodes:
            y[node_id] = float(row["label"])
            labeled_mask[node_id] = True

    # Split labeled nodes into train (80%) and test (20%)
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]
    perm = labeled_indices[torch.randperm(len(labeled_indices))]

    split = int(0.8 * len(perm))
    train_indices = perm[:split]
    test_indices = perm[split:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.labeled_mask = labeled_mask

    print(f"[OK] Loaded graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"     Features: {x.shape[1]} per node")
    print(f"     Labeled: {labeled_mask.sum().item()} / {num_nodes}")
    print(f"     Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")
    print(f"     Fraud ratio: {y[labeled_mask].sum().item():.0f} / {labeled_mask.sum().item()}")

    return data