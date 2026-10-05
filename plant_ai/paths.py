"""Central path configuration.

Every path in this project is resolved here so that scripts do not need to hardcode a
machine-specific absolute path. Resolution order for each:

1. environment variable (e.g. ``STRAWBERRY_PROJECT``)
2. a sensible default derived from this file's own location

The project directory is derived from ``__file__``, so a fresh clone works wherever it is
placed. The source dataset (``union_dataset``) lives OUTSIDE this project and must be pointed
at explicitly via ``STRAWBERRY_SOURCE_DATASET`` if it is not a sibling directory.

Override example (PowerShell)::

    $env:STRAWBERRY_SOURCE_DATASET = "E:\\datasets\\union_dataset"
"""

import os
from pathlib import Path

# This file lives at <project>/paths.py
PROJECT = Path(os.environ.get("STRAWBERRY_PROJECT", Path(__file__).resolve().parent))

DATASET = PROJECT / "dataset"
MANIFEST = DATASET / "manifest.csv"

EVAL_DIR = PROJECT / "eval"
DEMO_IMAGES = PROJECT / "demo_images"
SCREENSHOTS = PROJECT / "screenshots"
HEALTHY_CLEAN = PROJECT / "healthy_clean"
BINARY_DATASET = PROJECT / "binary_dataset"
PEST_AUDIT = PROJECT / "pest_audit"

BACKEND_DIR = PROJECT / "backend"
BACKEND_WEIGHTS = BACKEND_DIR / "best.pt"
DISEASE_INFO = BACKEND_DIR / "disease_info.json"
FRONTEND_DIR = PROJECT / "frontend"

# Vendored local clone of ultralytics/yolov5 (see README section 1)
YOLOV5_DIR = PROJECT / "yolov5"

# Run outputs
RUNS_DIR = YOLOV5_DIR / "runs"
FILTER_WEIGHTS = RUNS_DIR / "train-cls" / "filter" / "weights" / "best.pt"
STRAWBERRY9_WEIGHTS = RUNS_DIR / "train-cls" / "strawberry9" / "weights" / "best.pt"

# The shipped checkpoint: label-cleaned retrain.
SHIPPED_RUN = "strawberry9clean"
SHIPPED_WEIGHTS = RUNS_DIR / "train-cls" / SHIPPED_RUN / "weights" / "best.pt"
SHIPPED_EVAL_TAG = "ep100clean"

# Source corpus, outside the project. Defaults to a sibling of the project directory.
SOURCE_DATASET = Path(
    os.environ.get("STRAWBERRY_SOURCE_DATASET", PROJECT.parent / "union_dataset")
)
UNION_DATASET = SOURCE_DATASET


def require(path: Path, what: str) -> Path:
    """Return `path` if it exists, else raise a message that says how to fix it."""
    if not path.exists():
        raise SystemExit(
            f"[paths] {what} not found at: {path}\n"
            f"        Set the relevant environment variable or fix the layout. "
            f"See README section 1 (Environment setup)."
        )
    return path


def check_environment():
    """Report what is present and what is missing. Used by validate_dataset.py and start_servers.py."""
    items = [
        ("project", PROJECT, True),
        ("dataset", DATASET, True),
        ("manifest.csv", MANIFEST, True),
        ("backend weights", BACKEND_WEIGHTS, True),
        ("disease_info.json", DISEASE_INFO, True),
        ("yolov5 clone", YOLOV5_DIR, False),
        ("filter weights", FILTER_WEIGHTS, False),
        ("source dataset", SOURCE_DATASET, False),
    ]
    missing_required = []
    for name, p, required in items:
        ok = p.exists()
        if required and not ok:
            missing_required.append(f"{name} ({p})")
        print(f"  [{'ok' if ok else '--'}] {name:18s} {p}")
    return missing_required
