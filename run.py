"""
HCPG-GNN Auditor — Application Entry Point
============================================
Run the full application with:
    python run.py

This starts the FastAPI backend on http://localhost:8000
and serves the frontend dashboard at http://localhost:8000/app
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from backend.config import settings


def _preflight_checks():
    """Warn the user about missing model artifacts before the server starts."""
    models_dir = Path(__file__).parent / "models"
    checkpoint  = models_dir / "best_hgt_model.pt"
    metrics     = models_dir / "model_metrics.json"

    all_ok = True
    if not checkpoint.exists():
        print("  ⚠  WARNING: models/best_hgt_model.pt not found!")
        print("     The GNN model is NOT loaded — running in rule-based mode only.")
        print("     To train:  python models/train_model.py")
        all_ok = False
    else:
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"  ✓  Model checkpoint found ({size_mb:.1f} MB)")

    if not metrics.exists():
        print("  ⚠  WARNING: models/model_metrics.json not found!")
        print("     The dashboard will show N/A for all performance metrics.")
        all_ok = False
    else:
        import json
        try:
            with open(metrics) as f:
                m = json.load(f)
            f1 = m.get("f1_macro", m.get("f1_score", "?"))
            auc = m.get("AUC_ROC_macro", m.get("auc_roc", "?"))
            print(f"  ✓  Metrics file found — F1={f1:.4f}, AUC={auc:.4f}" if isinstance(f1, float) else "  ✓  Metrics file found")
        except Exception:
            print("  ⚠  WARNING: Could not parse model_metrics.json")
            all_ok = False

    if all_ok:
        print("  ✓  All checks passed — full GNN inference available")
    print()


def main():
    print("=" * 60)
    print("  HCPG-GNN Smart Contract Vulnerability Detector v4.0")
    print("=" * 60)
    print()
    _preflight_checks()
    print(f"  Backend API : http://localhost:{settings.PORT}")
    print(f"  Dashboard   : http://localhost:{settings.PORT}/app")
    print(f"  API Docs    : http://localhost:{settings.PORT}/docs")
    print(f"  Health Check: http://localhost:{settings.PORT}/health")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    uvicorn.run(
        "backend.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
