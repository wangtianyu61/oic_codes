"""Repository paths shared by experiment scripts and notebooks."""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'src' / 'datasets'
OUTPUT_ROOT = REPO_ROOT / 'output'


def output_path(*parts):
    """Return an output file path and create its parent directories."""
    path = OUTPUT_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(*parts):
    """Return a figure path under output/figures, creating its parents."""
    return output_path('figures', *parts)
