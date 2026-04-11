from pathlib import Path

# Project root is parent of src/
PROJECT_ROOT: Path = Path(__file__).parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
