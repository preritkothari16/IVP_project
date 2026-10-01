"""
Step 1a: build a balanced binary dataset (strawberry vs not-strawberry) for the
filter used to clean the contaminated `healthy` folder.

Fixes the earlier bug where a flat 2000-image cap truncated the positive list
before it ever reached the Strawberry___* folders. Here we sample a per-class
quota so every positive and negative folder is represented.

Positives: the 9 trusted strawberry folders.
Negatives: confirmed non-strawberry folders, capped and sampled evenly.
15% stratified holdout for the accuracy gate.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\ivp\union_dataset")
OUT = Path(r"D:\ivp\plant_ai\binary_dataset")
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png"}
TARGET_POS = 2000
TARGET_NEG = 2000
VAL_FRAC = 0.15

POSITIVE_CLASSES = [
    "angular_leafspot",
    "anthracnose_fruit_rot",
    "blossom_blight",
    "gray_mold",
    "leaf_spot",
    "powdery_mildew_fruit",
    "powdery_mildew_leaf",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
]

# Confirmed non-strawberry (verified by contact-sheet review in STEP A).
NEGATIVE_CLASSES = [
    "calciumdeficiency",
    "nutrient_deficiency",
    "grey_mold",
    "botrytis_cinerea",
    "burn",
    "marginal_leaf_necrosis",
]

# Extra clearly-non-strawberry folders used to broaden negative diversity.
NEGATIVE_EXTRA = [
    "esca",
    "cherry_leaf_spot",
    "corn_downy_mildew",
    "coccomyces_of_pome_fruits",
    "pear_blister_mite",
    "aphid",
    "thrips",
    "whitefly",
    "mealybug",
    "scale",
    "spider_mite",
    "leaf_miners",
    "mosaic_virus",
    "yellow_leaves",
    "edema",
    "sooty_mold",
]


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def even_sample(items, total_cap, rng):
    """Take up to total_cap items, spread across the list order-shuffled per class."""
    if len(items) <= total_cap:
        return list(items)
    return rng.sample(items, total_cap)


def main():
    rng = random.Random(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "train" / "strawberry").mkdir(parents=True)
    (OUT / "train" / "not_strawberry").mkdir(parents=True)
    (OUT / "val" / "strawberry").mkdir(parents=True)
    (OUT / "val" / "not_strawberry").mkdir(parents=True)

    manifest = []

    # ---- positives: equal per-class quota ----
    pos_pool = {c: list_images(ROOT / c) for c in POSITIVE_CLASSES}
    avail = sum(len(v) for v in pos_pool.values())
    quota = TARGET_POS // len(POSITIVE_CLASSES)
    print("POSITIVE per-class quota:", quota)
    pos_rows = []
    for c in POSITIVE_CLASSES:
        files = pos_pool[c]
        take = even_sample(files, quota, rng)
        print(f"  {c:28s} available={len(files):5d} taken={len(take):5d}")
        pos_rows += [(f, "strawberry", c) for f in take]
    print(f"  TOTAL positive = {len(pos_rows)} (pool {avail})")

    # ---- negatives: equal per-class quota across confirmed + extra folders ----
    neg_classes = NEGATIVE_CLASSES + NEGATIVE_EXTRA
    neg_pool = {c: list_images(ROOT / c) for c in neg_classes if (ROOT / c).is_dir()}
    neg_quota = max(1, TARGET_NEG // len(neg_pool))
    print("\nNEGATIVE per-class quota:", neg_quota)
    neg_rows = []
    for c, files in neg_pool.items():
        take = even_sample(files, neg_quota, rng)
        print(f"  {c:28s} available={len(files):5d} taken={len(take):5d}")
        neg_rows += [(f, "not_strawberry", c) for f in take]
    print(f"  TOTAL negative = {len(neg_rows)}")

    rows = pos_rows + neg_rows
    rng.shuffle(rows)

    counts = defaultdict(int)
    for path, label, src in rows:
        bucket = "val" if rng.random() < VAL_FRAC else "train"
        counts[(bucket, label)] += 1
        dest = OUT / bucket / label / f"{src}__{path.name}"
        shutil.copy2(path, dest)
        manifest.append(
            {
                "path": str(dest.relative_to(OUT)).replace("\\", "/"),
                "label": label,
                "source_folder": src,
                "is_plantvillage": src.startswith("Strawberry___"),
                "split": bucket,
            }
        )

    print("\n--- binary dataset split ---")
    for bucket in ("train", "val"):
        for label in ("strawberry", "not_strawberry"):
            print(f"  {bucket:5s} {label:15s} {counts[(bucket, label)]:5d}")
    total = sum(counts.values())
    print(f"  TOTAL {total}")

    with (OUT / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nmanifest -> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
