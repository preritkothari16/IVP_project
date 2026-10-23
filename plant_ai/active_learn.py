"""Active-learning sampling: identify high-uncertainty images for expert review.

Scores all images (or a specified split) through the live API using MC-dropout,
then ranks by uncertainty and exports a CSV of candidates for relabeling.

Usage:
    python active_learn.py --split test       # score TEST split
    python active_learn.py --split train      # score TRAIN split
    python active_learn.py --limit 50         # cap number of rows scored
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "dataset" / "manifest.csv"
OUT = PROJECT / "active_learn_candidates.csv"

API_BASE = "http://127.0.0.1:8000"


def predict_image(path: Path) -> dict:
    b = "----activelearning"
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
        f'filename="{path.name}"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()

    req = urllib.request.Request(
        f"{API_BASE}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max rows to score")
    ap.add_argument("--out", default=str(OUT), help="output CSV path")
    args = ap.parse_args()

    rows = [
        r
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
        if r["split"] == args.split and r["is_dup_copy"] == "False"
    ]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Scoring {len(rows)} images from split '{args.split}' ...")
    out_rows = []

    for i, r in enumerate(rows, 1):
        path = PROJECT / "dataset" / r["path"]
        try:
            d = predict_image(path)
        except Exception as e:
            print(f"  [{i}/{len(rows)}] ERROR {path}: {e}")
            continue

        top_class = d["top_class"]
        conf = d["confidence"]
        uncertainty = d.get("uncertainty")
        abstain = d.get("abstain", False)
        abstained_class = d.get("abstained_class")

        out_rows.append(
            {
                "path": r["path"],
                "true_class": r["final_class"],
                "predicted": top_class,
                "confidence": round(conf, 4),
                "uncertainty": round(uncertainty, 4) if uncertainty is not None else "",
                "abstain": abstain,
                "abstained_class": abstained_class or "",
                "top3": "|".join(f"{a['class_name']}:{a['confidence']:.3f}" for a in d.get("top3", [])),
                "correct": top_class == r["final_class"],
                "message": d.get("message") or "",
            }
        )

        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] ...")

    # Sort by uncertainty (higher = more uncertain)
    out_rows.sort(key=lambda x: float(x["uncertainty"]) if x["uncertainty"] else -1, reverse=True)

    fieldnames = [
        "path",
        "true_class",
        "predicted",
        "confidence",
        "uncertainty",
        "abstain",
        "abstained_class",
        "top3",
        "correct",
        "message",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {args.out}")
    print(f"\nTop 10 most uncertain:")
    for r in out_rows[:10]:
        uc = r["uncertainty"] if r["uncertainty"] != "" else "N/A"
        print(
            f"  {r['uncertainty'] if r['uncertainty'] else 'N/A':>8s} | "
            f"{r['true_class']:24s} -> {r['predicted']:24s} | conf={r['confidence']} | "
            f"{r['path']}"
        )

    # Summary stats
    n_abstain = sum(1 for r in out_rows if r["abstain"])
    n_incorrect = sum(1 for r in out_rows if r["correct"] == "False")
    print(f"\nSummary: {n_abstain} abstained, {n_incorrect} incorrect out of {len(out_rows)} scored")


if __name__ == "__main__":
    main()