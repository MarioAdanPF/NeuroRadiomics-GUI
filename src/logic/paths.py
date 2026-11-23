"""Rutas del repositorio, de la caché y del intérprete de Python del backend."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LUT_PATH = REPO_ROOT / "FreeSurferColorLUT.csv"
MODELS_DIR = REPO_ROOT / "models"
LOGOS_DIR = REPO_ROOT / "assets" / "logos"
EXTRACTOR_SCRIPT = REPO_ROOT / "legacy_service" / "extractor.py"
DEFAULT_OUTPUT_DIR = Path.home() / "NeuroRadiomics"

# Variables de entorno que pueden indicar el Python 3.9 del backend (PY_LEGACY es la original)
BACKEND_ENV_VARS = ("NEURORADIOMICS_BACKEND_PYTHON", "PY_LEGACY")


def cache_dir() -> Path:
    """Carpeta para los archivos temporales (conversiones de .mgz a .nii.gz)."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = base / "neuroradiomics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_backend_python() -> Path | None:
    """
    Busca el Python del backend: primero en las variables de entorno y después en el entorno
    virtual .venv-backend del repositorio (Linux/macOS o Windows).
    """
    candidates = [Path(os.environ[var]) for var in BACKEND_ENV_VARS if os.environ.get(var)]
    venv = REPO_ROOT / ".venv-backend"
    candidates += [venv / "bin" / "python", venv / "Scripts" / "python.exe"]
    return next((path for path in candidates if path.is_file()), None)
