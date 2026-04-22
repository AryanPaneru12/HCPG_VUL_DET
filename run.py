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


def main():
    print("=" * 60)
    print("  HCPG-GNN Smart Contract Vulnerability Detector v4.0")
    print("=" * 60)
    print()
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
