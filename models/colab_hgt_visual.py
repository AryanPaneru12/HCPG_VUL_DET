"""
Colab-ready training script for HCPG-GNN smart contract vulnerability detection.

Usage (Google Colab):
1) Upload this file and run all cells.
2) Optionally enable GPU from Runtime > Change runtime type > T4/A100.
3) Download `model_metrics.json` and `best_hgt_model.pt` for backend usage.

Architecture: Enhanced HGT-v4 with GATv2Conv + Residual Connections
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
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-3
    hidden_dim: int = 128
    heads: int = 4
    num_layers: int = 3
    classes: int = 5
    input_dim: int = 32
    dropout: float = 0.15
    num_samples: int = 3000


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ResidualGATBlock(nn.Module):
    """GATv2 convolution with residual connection and layer norm."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.gat = GATv2Conv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        residual = x
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return F.gelu(x)


class HGTVulnerabilityDetector(nn.Module):
    """
    Enhanced Heterogeneous Graph Transformer for HCPG-based
    smart-contract vulnerability detection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 heads: int = 4, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gat_blocks = nn.ModuleList([
            ResidualGATBlock(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        x = self.encoder(x)
        for block in self.gat_blocks:
            x = block(x, edge_index)
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True).values
        x = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(x)


# ============================================================================
# DATASET GENERATION
# ============================================================================

VULN_SIGNATURES = {
    0: {"call_before_state": 1.0, "external_call": 1.0, "state_write": 0.8, "payable": 0.7, "no_guard": 0.9},
    1: {"no_modifier": 1.0, "privileged_fn": 0.9, "selfdestruct": 0.6, "owner_pattern": 0.3, "external": 1.0},
    2: {"arithmetic_op": 1.0, "no_safemath": 0.9, "old_solidity": 0.8, "unchecked": 0.7, "loop": 0.5},
    3: {"low_level_call": 1.0, "no_check": 0.9, "send_pattern": 0.8, "transfer_miss": 0.7, "silent": 0.6},
    4: {"price_state": 0.9, "bid_pattern": 1.0, "approval": 0.7, "front_run": 0.9, "no_commit": 0.8},
}


def _build_feature(node_idx, total_nodes, vuln_class, is_vuln, dim, rng):
    feat = rng.standard_normal(dim).astype(np.float32) * 0.3
    feat[0] = 1.0 if node_idx < 3 else 0.0
    feat[1] = 1.0 if 3 <= node_idx < total_nodes // 2 else 0.0
    feat[2] = 1.0 if node_idx >= total_nodes // 2 else 0.0
    feat[3] = float(node_idx) / max(total_nodes, 1)
    for i in range(4, 16):
        feat[i] = rng.random() * 0.5

    if is_vuln:
        sigs = list(VULN_SIGNATURES[vuln_class].values())
        for j, val in enumerate(sigs):
            d = 16 + vuln_class * 3 + (j % 3)
            if d < dim:
                feat[d] += val * (0.7 + rng.random() * 0.3)
        if node_idx < 5:
            feat[16 + vuln_class * 3] += 0.5 * (1.0 + rng.random() * 0.2)
    else:
        for j in range(16, dim):
            feat[j] *= 0.2
    return feat


def generate_dataset(cfg: TrainConfig) -> List[Data]:
    rng = np.random.default_rng(cfg.input_dim)
    dataset = []
    per_class = cfg.num_samples // (cfg.classes + 1)
    clean = cfg.num_samples - per_class * cfg.classes

    for vc in range(cfg.classes):
        for _ in range(per_class):
            nn_nodes = rng.integers(25, 90)
            ne = int(nn_nodes * rng.uniform(1.8, 3.5))
            x = [_build_feature(i, nn_nodes, vc, True, cfg.input_dim, rng) for i in range(nn_nodes)]
            x = torch.tensor(np.stack(x), dtype=torch.float)
            src = torch.randint(0, nn_nodes, (ne,))
            dst = torch.randint(0, nn_nodes, (ne,))
            seq_src = torch.arange(0, min(nn_nodes - 1, 20))
            seq_dst = seq_src + 1
            edge_index = torch.stack([torch.cat([src, seq_src]), torch.cat([dst, seq_dst])], dim=0)
            y = torch.zeros(cfg.classes, dtype=torch.float)
            y[vc] = 1.0
            if rng.random() < 0.15:
                sec = rng.integers(0, cfg.classes)
                if sec != vc:
                    y[sec] = 1.0
            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    for _ in range(clean):
        nn_nodes = rng.integers(20, 70)
        ne = int(nn_nodes * rng.uniform(1.5, 3.0))
        x = [_build_feature(i, nn_nodes, 0, False, cfg.input_dim, rng) for i in range(nn_nodes)]
        x = torch.tensor(np.stack(x), dtype=torch.float)
        edge_index = torch.stack([torch.randint(0, nn_nodes, (ne,)), torch.randint(0, nn_nodes, (ne,))], dim=0)
        dataset.append(Data(x=x, edge_index=edge_index, y=torch.zeros(cfg.classes, dtype=torch.float)))

    rng.shuffle(dataset)
    return dataset


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ============================================================================
# TRAINING
# ============================================================================

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
    print(f"Device: {device}")

    data = generate_dataset(cfg)
    n = len(data)
    tr = int(0.8 * n)
    va = int(0.9 * n)
    train_loader = DataLoader(data[:tr], batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(data[tr:va], batch_size=cfg.batch_size)
    test_loader = DataLoader(data[va:], batch_size=cfg.batch_size)

    model = HGTVulnerabilityDetector(
        cfg.input_dim, cfg.hidden_dim, cfg.classes, cfg.heads, cfg.num_layers, cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-5)
    crit = FocalBCELoss()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        scheduler.step()
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

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color="#00e5ff")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Focal BCE Loss")
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, color="#22c55e")
    plt.title("Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.axhline(y=0.95, color="#ef4444", linestyle="--", alpha=0.5, label="95% target")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    with open("model_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "config": {
            "input_dim": cfg.input_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_classes": cfg.classes,
            "heads": cfg.heads,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
        },
        "metrics": metrics,
    }, "best_hgt_model.pt")

    print("Saved: best_hgt_model.pt, model_metrics.json, training_curves.png")
    print("Final metrics:", metrics)
    if metrics["accuracy"] >= 0.95:
        print(f"✅ TARGET MET: accuracy = {metrics['accuracy']:.4f}")
    else:
        print(f"⚠ Target not met: accuracy = {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
