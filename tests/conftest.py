"""
Pytest Fixtures and Configuration
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app import app


@pytest.fixture
def client():
    """Provide a FastAPI test client."""
    return TestClient(app)
