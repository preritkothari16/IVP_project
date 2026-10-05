"""Score every image in dataset/test through the live /predict endpoint."""
import paths  # central path configuration; see paths.py
import csv
import json
import threading
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


URL = "http://127.0.0.1:8000/predict"
ROOT = paths.DATASET
OUT = paths.EVAL_DIR / "lowconf"
OUT.mkdir(parents=True, exist_ok=True)
THRESH = 0.6
WORKERS = 4

files = sorted(p for p in (ROOT / "test").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(f"scanning {len(files)} test images via {URL}\n")


def probe(p):
    try:
        with Image.open(p) as im:
            fmt = im.format or "JPEG"
        data = p.read_bytes()
    except Exception as e:
        return {"file": p.name, "true": p.parent.name, "error": str(e)}
    b = f"----{uuid.uuid4().hex}"
    ctype = "image/png" if fmt == "PNG" else "image/jpeg"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{p.name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except Exception as e:
        return {"file": p.name, "true": p.parent.name, "error": str(e)}
    return {
        "file": p.name,
        "true": p.parent.name,
        "top_class": d.get("top_class"),
        "confidence": round(float(d.get("confidence", 0)), 4),
        "low_confidence": bool(d.get("low_confidence")),
        "correct": d.get("top_class") == p.parent.name,
        "top3": "; ".join(f"{a['class_name']}={a['confidence']:.4f}" for a in (d.get("top3") or [])),
    }


results = []
done = threading.Semaphore(0)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for r in ex.map(probe, files):
        results.append(r)

errs = [r for r in results if "error" in r]
if errs:
    print(f"!! {len(errs)} images failed:")
    for e in errs[:10]:
        print("   ", e["file"], e["error"])

ok = [r for r in results if "error" not in r]
with (OUT / "predict_test_set.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "true", "top_class", "confidence",
                                       "low_confidence", "correct", "top3"])
    w.writeheader()
    for r in sorted(ok, key=lambda x: (x["true"], x["file"])):
        w.writerow(r)

print(f"scored {len(ok)}/{len(files)}  ->  {OUT / 'predict_test_set.csv'}")

# cross-check against the training-eval output
ref = {Path(r["path"]).name: r["pred"] for r in
       csv.DictReader(open(paths.EVAL_DIR / "ep28_corrected" / "predictions.csv", encoding="utf-8"))}
shared = [r for r in ok if r["file"] in ref]
mismatch = [r for r in shared if r["top_class"] != ref[r["file"]]]
print(f"\nserved model vs training eval: {len(shared)} overlapping, {len(mismatch)} top-1 mismatches")
for m in mismatch[:8]:
    print(f"   {m['file']}: /predict={m['top_class']}  eval={ref[m['file']]}")
