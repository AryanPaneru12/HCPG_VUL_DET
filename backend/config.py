"""
Application Configuration & Settings
======================================
Centralised configuration loaded from environment variables and .env files.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings — immutable after creation."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # Model
    MODEL_FILENAME: str = "best_hgt_model.pt"
    METRICS_FILENAME: str = "model_metrics.json"

    # Server
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"

    # API
    API_VERSION: str = "4.0.0"
    API_TITLE: str = "HCPG-GNN Vulnerability Detection API"
    MODEL_NAME: str = "HGT-v4 (Heterogeneous Graph Transformer)"

    # CORS
    CORS_ORIGINS: list = ("*",)

    @property
    def model_path(self) -> Path:
        return self.MODELS_DIR / self.MODEL_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.MODELS_DIR / self.METRICS_FILENAME


settings = Settings()
