"""
Start the FastAPI backend and exercise it end-to-end:
  GET  /health    -> expect model_ready: true
  GET  /classes
  POST /predict   -> one real TEST image per class, plus bad-input cases
Results are printed as a table of actual vs predicted.
"""

import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset"
PORT = 8022
BASE = f"http://127.0.0.1:{PORT}"


def multipart(content, filename, ctype="image/jpeg"):
    b = "----b0undary"
    body = b"".join(
        [
            f"--{b}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            content,
            f"\r\n--{b}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={b}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return r.status, json.load(r)


def post(path, content, filename, ctype="image/jpeg"):
    body, hdr = multipart(content, filename, ctype)
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": hdr})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(PROJECT / "backend"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        time.sleep(20)
        st, h = get("/health")
        print(f"GET /health -> {st}")
        print(json.dumps(h, indent=2))
        print(f"\nmodel_ready = {h['model_ready']}   (must be True to continue)\n")
        if not h["model_ready"]:
            print("ABORT: model not ready")
            return

        st, c = get("/classes")
        print(f"GET /classes -> {st}  count={c['count']}")
        print("  detects      :", c["detects"])
        print("  does_not_detect:", c["does_not_detect"], "\n")

        # one real TEST image per class
        import csv

        rows = [r for r in csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8"))
                if r["split"] == "test" and r["is_dup_copy"] == "False"]
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["final_class"]].append(r)

        print("=== /predict on one real image per class (TEST split) ===")
        print(f"{'#':>2s} {'actual':22s} {'predicted':22s} {'ok':>4s} {'conf':>7s} {'sev':>7s} {'lowconf':>8s}")
        n_ok = 0
        n = 0
        for cls in sorted(by_cls):
            r = by_cls[cls][0]
            data = (DATASET / r["path"]).read_bytes()
            n += 1
            st, js = post("/predict", data, Path(r["path"]).name)
            if st != 200:
                print(f"{n:2d} {cls:22s} HTTP {st} {js}")
                continue
            ok = js["top_class"] == cls
            n_ok += ok
            print(f"{n:2d} {cls:22s} {js['top_class']:22s} {'YES' if ok else 'NO':>4s} "
                  f"{js['confidence']:7.4f} {js['disease_info']['severity']:>7s} "
                  f"{str(js['low_confidence']):>8s}")
        print(f"\n  correct {n_ok}/{n}\n")

        # show one full response body for the UI/frontend contract
        r = by_cls[sorted(by_cls)[0]][0]
        st, js = post("/predict", (DATASET / r["path"]).read_bytes(), "sample.jpg")
        print("=== full /predict response shape (1st class) ===")
        print(json.dumps(js, indent=2)[:2200])
        print("\n=== top3 field types ===")
        print("  top3[0] =", json.dumps(js["top3"][0]))

        # ---- bad inputs ----
        print("\n=== bad-input handling ===")
        txt = b"this is definitely not an image, just plain text pretending to be a jpg" * 3
        st, js = post("/predict", txt, "fake.jpg")
        print(f"  .txt renamed .jpg  -> HTTP {st}  {js}")

        big = b"\xff\xd8\xff\xe0" + b"\x00" * (20 * 1024 * 1024)
        st, js = post("/predict", big, "huge.jpg")
        print(f"  20 MB file         -> HTTP {st}  {str(js)[:130]}")

        gray = np.full((400, 400, 3), 127, np.uint8)
        ok, buf = cv2.imencode(".jpg", gray)
        st, js = post("/predict", buf.tobytes(), "blank.jpg")
        print(f"  featureless grey   -> HTTP {st}  top={js.get('top_class')} "
              f"conf={js.get('confidence')} low={js.get('low_confidence')}")

        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        st, js = post("/predict", png, "x.png", "image/png")
        print(f"  corrupt png        -> HTTP {st}  {str(js)[:130]}")

        st, js = post("/predict", b"\x00\x01\x02", "x.txt", "text/plain")
        print(f"  text/plain upload  -> HTTP {st}  {str(js)[:130]}")
    finally:
        proc.terminate()
        time.sleep(2)
        try:
            out = proc.stdout.read()
        except Exception:
            out = ""
        tail = [l for l in (out or "").strip().splitlines()[-6:]]
        if tail:
            print("\n--- server log tail ---")
            print("\n".join(tail))


if __name__ == "__main__":
    import numpy as np

    main()
