import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `training/` modules import each other as top-level packages (e.g. `from data...`,
# `from train_utils...`), and scripts/ is imported by the archive tests.
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "training"), os.path.join(REPO_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
