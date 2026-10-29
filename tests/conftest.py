import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "classes_micro.yaml"
MACRO_CONFIG = ROOT / "configs" / "classes_macro.yaml"
# Mineral-name label space used to exercise aliases, merges and renames.
MINERAL_CONFIG = ROOT / "tests" / "fixtures" / "classes_minerals.yaml"
