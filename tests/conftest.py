"""Ensures the project root is importable regardless of how pytest is invoked."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
