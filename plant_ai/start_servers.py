"""Start the backend and the Vite dev server, and keep them running in the background."""

import paths  # central path configuration; see paths.py
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = str(paths.PROJECT)

# npm is not always on PATH for GUI-launched processes (it is a .cmd shim on Windows and the
# PATH used by the desktop differs from the one in a terminal). Resolve it robustly:
# explicit override, then PATH lookup, then the standard install location, then bare "npm".
NPM = os.environ.get("NPM") or shutil.which("npm.cmd") or shutil.which("npm") or shutil.which("npm.exe")
if NPM is None:
    fallback = Path(r"C:\Program Files\nodejs\npm.cmd")
    NPM = str(fallback) if fallback.exists() else "npm"

BACKEND_PORT = os.environ.get("BACKEND_PORT", "8000")
FRONTEND_PORT = os.environ.get("FRONTEND_PORT", "5173")

backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", BACKEND_PORT],
    cwd=os.path.join(ROOT, "backend"),
    stdout=open(os.path.join(ROOT, "backend", "server.log"), "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
)
print("backend pid", backend.pid)

frontend = subprocess.Popen(
    [NPM, "run", "dev"],
    cwd=os.path.join(ROOT, "frontend"),
    stdout=open(os.path.join(ROOT, "frontend", "dev.log"), "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
print("frontend pid", frontend.pid)

time.sleep(14)
for label, url in ((f"backend /health", f"http://127.0.0.1:{BACKEND_PORT}/health"),
                   (f"frontend /", f"http://127.0.0.1:{FRONTEND_PORT}/")):
    try:
        r = urllib.request.urlopen(url, timeout=15)
        print(f"{label} -> {r.status}")
    except Exception as exc:
        print(f"{label} -> FAILED {exc}")

with open(os.path.join(ROOT, "pids.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"{backend.pid}\n{frontend.pid}\n")
print("pids written to", os.path.join(ROOT, "pids.txt"))
print(f"open http://127.0.0.1:{FRONTEND_PORT}/ in a browser")
