import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from api/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

TMDB_API_KEY: str = os.getenv("API_TMDB_KEY", "")
TMDB_READ_TOKEN: str = os.getenv("API_TMDB_READ_KEY", "")

# MODELS_PATH can be absolute or relative to project root
_models_env = os.getenv("MODELS_PATH", "models")
_models_path = Path(_models_env)
MODELS_PATH: Path = (_root / _models_path).resolve() if not _models_path.is_absolute() else _models_path.resolve()
