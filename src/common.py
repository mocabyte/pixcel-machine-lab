"""
common.py — shared helpers for all Pixcel Machine Lab scripts.

- DATA_DIR / MODEL_DIR / OUTPUT_DIR: paths that work no matter
  which folder you start a script from.
- get_device(): picks CUDA, Apple Silicon (MPS) or CPU for PyTorch.
- download_model(): fetches a model file once and caches it in models/.
"""

from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"


def data_path(name: str) -> Path:
    """Path to a file in data/. Fails early with a readable message."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return path


def output_path(name: str) -> Path:
    """Path for results. Creates output/ on first use."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name


def get_device() -> str:
    """'cuda' on NVIDIA GPUs, 'mps' on Apple Silicon, otherwise 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_model(url: str, filename: str | None = None) -> Path:
    """Download a model file into models/ the first time it is needed."""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / (filename or url.rsplit("/", 1)[-1])
    if not path.exists():
        print(f"Downloading {path.name} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path
