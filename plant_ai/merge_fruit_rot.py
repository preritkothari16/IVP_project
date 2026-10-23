"""Merge the 3 fruit-rot classes into a single `fruit_rot` class.

This addresses the root cause: anthracnose_fruit_rot, gray_mold, and
powdery_mildew_fruit are not reliably separable from a single photo.
Merging them means the model stops trying to distinguish what is
biologically ambiguous, and the UI can surface the honest answer:
"this is a fruit rot — consult an agronomist."

Usage:
    python merge_fruit_rot.py [--dry-run]
"""
import argparse
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "dataset" / "manifest.csv"
OUT = PROJECT / "dataset" / "manifest_merged.csv"

FRUIT_ROT_CLASSES = {
    "anthracnose_fruit_rot",
    "gray_mold",
    "powdery_mildew_fruit",
}
MERGED_CLASS = "fruit_rot"


def merge(path: Path = MANIFEST, out: Path = OUT, dry_run: bool = False):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    merged_rows = []
    counts = {c: 0 for c in FRUIT_ROT_CLASSES}
    for r in rows:
        new_r = dict(r)
        if r["final_class"] in FRUIT_ROT_CLASSES:
            counts[r["final_class"]] += 1
            new_r["final_class"] = MERGED_CLASS
        merged_rows.append(new_r)

    print(f"Original rows: {len(rows)}")
    print(f"  anthracnose_fruit_rot: {counts['anthracnose_fruit_rot']}")
    print(f"  gray_mold:             {counts['gray_mold']}")
    print(f"  powdery_mildew_fruit:  {counts['powdery_mildew_fruit']}")
    print(f"  -> merged into '{MERGED_CLASS}': {sum(counts.values())}")

    if not dry_run:
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"Wrote {out}")

    # Summary per split
    from collections import Counter
    for split in ("train", "val", "test"):
        c = Counter(r["final_class"] for r in merged_rows if r["split"] == split)
        print(f"  {split}: {dict(c)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in", dest="input", default=str(MANIFEST))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    merge(Path(args.input), Path(args.out), dry_run=args.dry_run)