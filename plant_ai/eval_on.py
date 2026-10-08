"""Run evaluate.py against an arbitrary dataset root + manifest, without editing paths.py.

evaluate.py resolves DATASET from paths.py at import time and reads <DATASET>/manifest.csv.
This shim patches both so an alternative test set can be scored with the same code path, which
matters: a separate reimplementation would risk diverging from the shipped evaluation.
"""
import runpy
import shutil
import sys
from pathlib import Path

import paths

DATASET_ROOT = Path(sys.argv[1]).resolve()
MANIFEST = Path(sys.argv[2]).resolve()
TAG = sys.argv[3]
WEIGHTS = sys.argv[4]

target = DATASET_ROOT / "manifest.csv"
backup = DATASET_ROOT / "manifest.csv.bak"
moved = False
if target.exists():
    shutil.copy2(target, backup)
    moved = True
shutil.copy2(MANIFEST, target)

paths.DATASET = DATASET_ROOT
paths.MANIFEST = target

sys.argv = ["evaluate.py", WEIGHTS, TAG]
try:
    runpy.run_path(str(paths.PROJECT / "evaluate.py"), run_name="__main__")
finally:
    if moved:
        shutil.copy2(backup, target)
        backup.unlink()