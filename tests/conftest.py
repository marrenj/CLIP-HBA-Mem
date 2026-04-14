"""Pytest configuration: add CLIP-HBA/ to sys.path so pipeline modules are importable."""
import sys
from pathlib import Path

# Allow ``from functions.xxx import yyy`` without installing the package.
CLIP_HBA_DIR = Path(__file__).parent.parent / "CLIP-HBA"
if str(CLIP_HBA_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_HBA_DIR))
