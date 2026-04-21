"""
Colab-ready training script for HCPG-GNN smart contract vulnerability detection.

Usage (Google Colab):
1) Upload this file and run all cells.
2) Optionally enable GPU from Runtime > Change runtime type > T4/A100.
3) Download `model_metrics.json` and `best_hgt_model.pt` for backend usage.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 2e-3
    hidden_dim: int = 128
    heads: int = 4
    classes: int = 5
    input_dim: int = 16


class HGTStyleDetector(nn.Module):
    """
    Compact high-performing graph model for Colab experiments.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, heads: int):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.enc(x))
        x = F.relu(self.gat1(x, edge_index))
        x = self.norm(x)
        x = F.relu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)


def synthetic_hcpg_dataset(n: int = 1200, input_dim: int = 16, classes: int = 5) -> List[Data]:
    """
    Synthetic fallback so the script always runs in Colab demos.
    Replace this with your real HCPG extraction pipeline for final experiments.
    """
    dataset: List[Data] = []
    for _ in range(n):
        nodes = np.random.randint(20, 80)
        x = torch.randn(nodes, input_dim)
        src = torch.randint(0, nodes, (nodes * 2,))
        dst = torch.randint(0, nodes, (nodes * 2,))
        edge_index = torch.stack([src, dst], dim=0)
        y = torch.randint(0, 2, (classes,), dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


def split_dataset(dataset: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
    random.shuffle(dataset)
    n = len(dataset)
    tr = int(0.7 * n)
    va = int(0.85 * n)
    return dataset[:tr], dataset[tr:va], dataset[va:]


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.batch)
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            y_prob.append(prob)
            y_pred.append(pred)
            y_true.append(b.y.cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_prob = np.vstack(y_prob)
    return {
        "accuracy": float(accuracy_score(y_true.flatten(), y_pred.flatten())),
        "precision": float(precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)),
        "recall": float(recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)),
        "f1_score": float(f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true.flatten(), y_prob.flatten())),
    }


def main():
    seed_everything(42)
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = synthetic_hcpg_dataset(n=1200, input_dim=cfg.input_dim, classes=cfg.classes)
    train_set, val_set, test_set = split_dataset(data)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    model = HGTStyleDetector(cfg.input_dim, cfg.hidden_dim, cfg.classes, cfg.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    train_losses, val_f1s = [], []
    best_state, best_f1 = None, -1.0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.batch)
            loss = crit(out, b.y)
            loss.backward()
            opt.step()
            running += loss.item()
        train_losses.append(running / max(1, len(train_loader)))
        val_metrics = evaluate(model, val_loader, device)
        val_f1s.append(val_metrics["f1_score"])
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}: loss={train_losses[-1]:.4f} val_f1={val_f1s[-1]:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s)
    plt.title("Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    with open("model_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    torch.save(model.state_dict(), "best_hgt_model.pt")

    print("Saved: best_hgt_model.pt, model_metrics.json, training_curves.png")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
