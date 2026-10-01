"""
Step 1b: score every image in the contaminated `healthy` folder with the trained
binary strawberry-vs-not filter, emit a CSV, and split the results into
keep / drop / uncertain. Also writes numbered 8x8 review sheets plus random
kept/dropped spot-check sheets.
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\ivp\plant_ai")
YOLOV5 = PROJECT / "yolov5"
WEIGHTS = YOLOV5 / "runs" / "train-cls" / "filter" / "weights" / "best.pt"
HEALTHY_DIR = Path(r"D:\ivp\union_dataset\healthy")
OUT = PROJECT / "healthy_clean"
SHEETS = OUT / "review_uncertain"
SPOT = OUT / "spotcheck"
IMG_SIZE = 224
KEEP_AT = 0.85
DROP_AT = 0.15
SEED = 42

sys.path.append(str(YOLOV5))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
except Exception:
    FONT = ImageFont.load_default()


def stamp(img, x, y, text):
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), text, font=FONT)
    w, h = bb[2] - bb[0] + 12, bb[3] - bb[1] + 10
    d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
    d.text((x + 6, y + 5), text, fill=(255, 240, 0), font=FONT)


def sheet(rows, out_path, cols=8, cell=256, tag=""):
    """rows = [(label, path)] ; one 8x8 page per 64 rows, page size varies to be unique."""
    pages = [rows[i : i + cols * 8] for i in range(0, len(rows), cols * 8)]
    written = []
    for pno, page in enumerate(pages):
        grid = Image.new("RGB", (cols * cell, 8 * cell), (20, 20, 20))
        for i, (label, path) in enumerate(page):
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    sc = cell / max(im.size)
                    im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
                    grid.paste(
                        im,
                        (
                            (i % cols) * cell + (cell - im.width) // 2,
                            (i // cols) * cell + (cell - im.height) // 2,
                        ),
                    )
            except Exception:
                pass
            cx, cy = (i % cols) * cell, (i // cols) * cell
            ImageDraw.Draw(grid).rectangle(
                [cx, cy, cx + cell - 1, cy + cell - 1], outline=(120, 120, 120), width=2
            )
            stamp(grid, cx + 2, cy + 2, label)
        canvas = Image.new("RGB", (cols * cell + 7 + pno, 8 * cell), (20, 20, 20))
        canvas.paste(grid, (0, 0))
        p = out_path.format(page=pno)
        canvas.save(p)
        written.append(str(p))
        print(f"    {p}  ({len(page)} thumbs)")
    return written


def main():
    device = select_device("")
    model = DetectMultiBackend(str(WEIGHTS), device=device, fuse=False)
    raw = model.names
    # yolov5 exposes ClassificationModel.names as {index: name}; normalise to an ordered list.
    names = [raw[i] for i in sorted(raw)] if isinstance(raw, dict) else list(raw)
    tf = classify_transforms(IMG_SIZE)
    model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
    straw_idx = names.index("strawberry")
    print(f"[filter] classes={names} strawberry_idx={straw_idx} device={device}\n")

    files = sorted(
        p for p in HEALTHY_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"[filter] scoring {len(files)} images from {HEALTHY_DIR}")

    rows, skipped = [], []
    for n, p in enumerate(files, 1):
        try:
            with Image.open(p) as im:
                im.load()
                rgb = im.convert("RGB")
        except Exception as exc:
            skipped.append((str(p), str(exc)))
            print(f"  SKIP (unreadable) {p.name}: {exc}")
            continue
        t = tf(np.asarray(rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            prob = F.softmax(model(t), dim=1)[0, straw_idx].item()
        bucket = "keep" if prob >= KEEP_AT else ("drop" if prob <= DROP_AT else "uncertain")
        rows.append({"path": str(p), "name": p.name, "p_strawberry": round(prob, 6), "bucket": bucket})
        if n % 200 == 0:
            print(f"  ...{n}/{len(files)}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "healthy_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "name", "p_strawberry", "bucket"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[filter] CSV -> {OUT / 'healthy_scores.csv'}")

    keeps = [r for r in rows if r["bucket"] == "keep"]
    drops = [r for r in rows if r["bucket"] == "drop"]
    unc = [r for r in rows if r["bucket"] == "uncertain"]
    print(f"[filter] KEEP={len(keeps)}  DROP={len(drops)}  UNCERTAIN={len(unc)}  SKIPPED={len(skipped)}")
    if unc:
        band = f"{unc[0]['p_strawberry']:.3f}..{unc[-1]['p_strawberry']:.3f}"
        print(f"[filter] uncertain p range: {band}")

    SHEETS.mkdir(parents=True, exist_ok=True)
    if unc:
        rng = random.Random(SEED)
        rng.shuffle(unc)
        labelled = [(f"U{i:03d} {r['p_strawberry']:.2f}", r["path"]) for i, r in enumerate(unc)]
        print(f"\n[filter] uncertain review sheets ({len(labelled)} thumbs):")
        sheet(labelled, str(SHEETS / "uncertain_page{page}.png"))

    SPOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for name, pool in (("dropped", drops), ("kept", keeps)):
        if not pool:
            continue
        pick = rng.sample(pool, min(32, len(pool)))
        labelled = [(f"{name[0].upper()}{i:02d} {r['p_strawberry']:.2f}", r["path"]) for i, r in enumerate(pick)]
        print(f"\n[filter] spot-check sample for {name} ({len(pick)} thumbs):")
        sheet(labelled, str(SPOT / f"{name}_sample_page{{page}}.png"))

    summary = {
        "weights": str(WEIGHTS),
        "n_scored": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "keep": len(keeps),
        "drop": len(drops),
        "uncertain": len(unc),
        "keep_at": KEEP_AT,
        "drop_at": DROP_AT,
    }
    (OUT / "clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[filter] summary -> {OUT / 'clean_summary.json'}")


if __name__ == "__main__":
    main()
