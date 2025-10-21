"""Pytest configuration for notebooks.

Ensures the local source tree is importable as `pydflt` when running notebooks without installing the package first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
