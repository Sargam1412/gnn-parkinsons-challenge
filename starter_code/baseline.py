import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from dgl.nn import GraphConv

torch.manual_seed(42)
np.random.seed(42)
dgl.seed(42)


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, activation=F.relu)
        self.conv2 = GraphConv(hidden_size, hidden_size, activation=F.relu)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, g, features):
        h = F.dropout(features, p=self.dropout, training=self.training)
        h = self.conv1(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return self.classifier(h)


def load_data():
    """
    Loads DGL-free pkl files and rebuilds DGL graphs at runtime from edge_index.
    This means the pkl files themselves have zero DGL dependency.
    """
    print("Loading data...")
    with open('../data/train_graph_free.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../data/test_graph_free.pkl', 'rb') as f:
        test_data = pickle.load(f)

    def rebuild_dgl_graph(d):
        src, dst = d["edge_index"]
        g = dgl.graph((src, dst), num_nodes=d["num_nodes"])
        d["graph"] = g
        return d

    train_data = rebuild_dgl_graph(train_data)
    test_data  = rebuild_dgl_graph(test_data)
    return train_data, test_data


def train_epoch(model, g, features, labels, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(logits[train_mask], 1)
    train_acc = (predicted == labels[train_mask]).float().mean()
    return loss.item(), train_acc.item()


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, predicted = torch.max(logits[mask], 1)
        accuracy = (predicted == labels[mask]).float().mean()
        f1_macro = f1_score(
            labels[mask].cpu().numpy(),
            predicted.cpu().numpy(),
            average='macro'
        )
    return accuracy.item(), f1_macro


def main():
    print("=" * 60)
    print("GNN Parkinson's Challenge - Baseline GCN Model")
    print("=" * 60)

    train_data, test_data = load_data()

    g          = train_data['graph']
    features   = train_data['features']
    labels     = train_data['labels']
    train_mask = train_data['train_mask']
    val_mask   = train_data['val_mask']

    print(f"\nDataset Statistics:")
    print(f"  Nodes: {g.num_nodes()}")
    print(f"  Edges: {g.num_edges()}")

    in_feats     = features.shape[1]
    hidden_size  = 64
    num_classes  = 2
    dropout      = 0.5
    lr           = 0.01
    weight_decay = 5e-4
    num_epochs   = 200

    print(f"\nModel: GCN")

    model     = GCNModel(in_feats, hidden_size, num_classes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\nTraining...")
    print("-" * 60)

    best_val_f1      = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        loss, train_acc = train_epoch(model, g, features, labels, train_mask, optimizer)
        val_acc, val_f1 = evaluate(model, g, features, labels, val_mask)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gcn_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= 50:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest Validation F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load('best_gcn_model.pt'))

    print("\nGenerating predictions...")
    test_g        = test_data['graph']
    test_features = test_data['features']
    test_node_ids = test_data['node_ids']

    model.eval()
    with torch.no_grad():
        test_logits = model(test_g, test_features)
        _, test_predictions = torch.max(test_logits, 1)

    submission = pd.DataFrame({
        'node_id':    test_node_ids,
        'prediction': test_predictions.cpu().numpy()
    })

    submission.to_csv('../submissions/gcn_submission.csv', index=False)
    print("Submission saved to submissions/gcn_submission.csv")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
