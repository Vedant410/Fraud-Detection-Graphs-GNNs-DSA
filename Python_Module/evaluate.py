from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = torch.sigmoid(logits) > 0.5

    # Only evaluate on LABELED test nodes
    mask = data.test_mask

    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Precision:", precision)
    print("Recall:", recall)  # MOST IMPORTANT
    print("F1-score:", f1)