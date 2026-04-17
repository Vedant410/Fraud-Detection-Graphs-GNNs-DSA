import torch
from gnn_model import FraudGNN
from load_data import load_data

data = load_data()

model = FraudGNN(in_channels=data.num_features, hidden=32)

# Class imbalance: weight the loss toward the minority class (fraud)
train_labels = data.y[data.train_mask]
fraud_ratio = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=fraud_ratio)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    # Only compute loss on LABELED training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")