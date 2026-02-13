"""Shared test configuration and fixtures."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force test-friendly environment variables.
os.environ.setdefault("HEDGE__PROJECT__ENVIRONMENT", "paper")
os.environ.setdefault("HEDGE__EXECUTION__PAPER_ONLY", "true")
