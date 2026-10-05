"""
Step 3a: build the final 9-class training dataset.

1. Collect images for the 9 final classes (healthy = 425 filtered + 456 PlantVillage).
2. Perceptual-hash (dHash) dedupe WITHIN and ACROSS classes, before splitting, so no
   near-duplicate can leak between train/val/test.
3. Stratified 70/20/10 split, guaranteeing >=2 val and >=2 test per class.
4. TRAIN-only class balancing: cap at 600/class, oversample classes under 40 up to 40
   (copies named *_dupN). Never oversample val/test.
5. Emit manifest.csv with per-image is_plantvillage evidence, and print per-class
   PlantVillage fraction per split.
"""

import paths  # central path configuration; see paths.py
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SRC = paths.SOURCE_DATASET
SCORES = paths.HEALTHY_CLEAN / 'healthy_scores.csv'
OUT = paths.DATASET
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png"}
DHASH_BITS = 64
NEAR_DUP_THRESHOLD = 5
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.20, 0.10
MIN_VAL, MIN_TEST = 2, 2
CAP_TRAIN = 600
MIN_TRAIN = 40
# Per-class cap on the TOTAL number of images (applied after dedupe, BEFORE splitting) so that a
# dominant class cannot dominate val/test either. leaf_scorch is capped because it is 100%
# PlantVillage and would otherwise skew the by-source evaluation.
CLASS_TOTAL_CAP = {"leaf_scorch": 600}

# final class -> list of source folders
CLASS_SOURCES = {
    "angular_leafspot": ["angular_leafspot"],
    "anthracnose_fruit_rot": ["anthracnose_fruit_rot"],
    "blossom_blight": ["blossom_blight"],
    "gray_mold": ["gray_mold"],
    "leaf_spot": ["leaf_spot"],
    "powdery_mildew_fruit": ["powdery_mildew_fruit"],
    "powdery_mildew_leaf": ["powdery_mildew_leaf"],
    "leaf_scorch": ["Strawberry___Leaf_scorch"],
    "healthy": ["Strawberry___healthy", "healthy"],  # 2nd is the FILTERED folder
}
FILTERED_FOLDER = "healthy"


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def dhash(path: Path, hash_size: int = 8) -> int:
    """64-bit difference hash: compares adjacent pixels, invariant to resize/small edits."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("unreadable")
    small = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in diff.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.bitwise_xor(a, b))


def looks_plantvillage(path: Path) -> bool:
    """DEPRECATED / NOT USED.

    A bright-uniform-background heuristic was trialled and rejected: calibrated against known
    PlantVillage controls (Strawberry___healthy / Strawberry___Leaf_scorch) it only reached 8.3%
    recall, because those studio shots are leaf-filled and not bright-bordered. is_plantvillage is
    therefore decided by source folder alone, which is exact. A visual audit of 64 kept ex-`healthy`
    images (healthy_clean/kept_sheet_20260929.png) confirmed all are field/garden shots, so none of
    them are PlantVillage-style.
    """
    return False


def main():
    rng = random.Random(SEED)

    keep_names = set()
    if SCORES.is_file():
        with SCORES.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["bucket"] == "keep":
                    keep_names.add(r["name"])
        print(f"filtered 'healthy' keep-list: {len(keep_names)} images")
    else:
        raise SystemExit(f"missing {SCORES}")

    # ---------- 1. collect ----------
    pool = []  # (class, source_folder, path)
    skipped = []
    for cls, folders in CLASS_SOURCES.items():
        for folder in folders:
            d = SRC / folder
            files = list_images(d)
            if folder == FILTERED_FOLDER:
                before = len(files)
                files = [p for p in files if p.name in keep_names]
                print(f"  {folder:26s} {before} -> {len(files)} after filter")
            for p in files:
                pool.append((cls, folder, p))
    print(f"\ncollected {len(pool)} images across {len(CLASS_SOURCES)} classes")

    # ---------- 2. perceptual-hash dedupe (within + across classes) ----------
    hashes, valid, hash_errs = [], [], []
    for cls, folder, p in pool:
        try:
            hashes.append(dhash(p))
            valid.append((cls, folder, p))
        except Exception:
            hash_errs.append(str(p))
    print(f"hashed {len(valid)} images ({len(hash_errs)} unreadable/skipped)")

    order = sorted(range(len(valid)), key=lambda i: (valid[i][0], valid[i][2].name))
    kept_idx, kept_h = [], []
    kh = np.zeros(0, dtype=np.uint64)
    exact_dup, near_dup, cross_class = [], [], []
    for i in order:
        h = np.uint64(hashes[i])
        if kh.size:
            d = hamming(kh, np.array([h], dtype=np.uint64))
            j = int(np.argmin(d))
            dist = int(d[j])
            if dist <= NEAR_DUP_THRESHOLD:
                k = kept_idx[j]
                same = valid[k][0] == valid[i][0]
                (exact_dup if dist == 0 else near_dup).append((valid[i][2], valid[i][0], valid[k][0], dist))
                if not same:
                    cross_class.append((valid[i][2], valid[i][0], valid[k][0], dist))
                continue
        kept_idx.append(i)
        kept_h.append(int(h))
        kh = np.append(kh, np.uint64(h))
    kept = [valid[i] for i in kept_idx]
    print(
        f"\ndedupe: kept {len(kept)}  removed exact={len(exact_dup)} near={len(near_dup)} "
        f"(cross-class conflicts={len(cross_class)})"
    )

    # ---------- 2b. per-class TOTAL cap (after dedupe, before splitting) ----------
    total_capped = []
    by_cls_dedup = defaultdict(list)
    for cls, folder, p in kept:
        by_cls_dedup[cls].append((cls, folder, p))
    kept = []
    for cls in sorted(CLASS_SOURCES):
        items = by_cls_dedup[cls]
        cap = CLASS_TOTAL_CAP.get(cls)
        if cap is not None and len(items) > cap:
            rng.shuffle(items)
            total_capped.append((cls, len(items) - cap, cap))
            items = items[:cap]
        kept += items
    if total_capped:
        print("\nTOTAL cap applied before split:")
        for cls, n, cap in total_capped:
            print(f"    {cls:24s} dropped {n:4d} -> capped at {cap} total")

    # ---------- 3. stratified split ----------
    by_cls = defaultdict(list)
    for cls, folder, p in kept:
        by_cls[cls].append((folder, p))
    split_rows = []
    for cls in sorted(CLASS_SOURCES):
        items = by_cls[cls]
        rng.shuffle(items)
        n = len(items)
        n_test = max(MIN_TEST, round(n * TEST_FRAC))
        n_val = max(MIN_VAL, round(n * VAL_FRAC))
        while n_test + n_val + 1 > n:  # ensure at least 1 train image
            if n_val > MIN_VAL:
                n_val -= 1
            elif n_test > MIN_TEST:
                n_test -= 1
            else:
                break
        n_train = n - n_val - n_test
        for label, chunk in (("train", items[:n_train]), ("val", items[n_train : n_train + n_val]),
                             ("test", items[n_train + n_val :])):
            for folder, p in chunk:
                split_rows.append(
                    {"class": cls, "folder": folder, "path": p, "split": label, "dup": False, "hash": 0}
                )
    # rebuild hashes properly
    pos = {(c, f, p): hashes[i] for i, (c, f, p) in enumerate(valid)}
    for r in split_rows:
        r["hash"] = pos[(r["class"], r["folder"], r["path"])]

    # ---------- 4. TRAIN-only balancing ----------
    tr_by_cls = defaultdict(list)
    for r in split_rows:
        if r["split"] == "train":
            tr_by_cls[r["class"]].append(r)
    added_dups, capped = [], []
    for cls, rows in tr_by_cls.items():
        real = [r for r in rows if not r["dup"]]
        if len(real) > CAP_TRAIN:
            rng.shuffle(real)
            removed = real[CAP_TRAIN:]
            capped.append((cls, len(removed)))
            keep_rows = real[:CAP_TRAIN]
            for r in rows:
                r["_drop"] = r in removed
            tr_by_cls[cls] = keep_rows
            rows = keep_rows
        n_real = len([r for r in rows if not r["dup"]])
        if n_real < MIN_TRAIN:
            base = [r for r in rows if not r["dup"]]
            n_dup = sum(1 for r in rows if r["dup"])
            k = 1
            while base and n_dup < MIN_TRAIN - n_real:
                src = base[k % len(base)]
                new = dict(src)
                new["dup"] = True
                new["dup_src"] = src["path"].name
                new["path"] = src["path"].with_name(f"{src['path'].stem}__dup{k}{src['path'].suffix}")
                rows.append(new)
                added_dups.append((cls, new["path"]))
                n_dup += 1
                k += 1
        tr_by_cls[cls] = rows

    # ---------- 5. write dataset ----------
    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    manifest = []
    all_rows = list(split_rows)
    for r in all_rows:
        r.pop("_drop", None)
    # rebuild from final per-class train rows (post cap/oversample) + val/test
    final_rows = []
    for cls in sorted(CLASS_SOURCES):
        final_rows += tr_by_cls[cls]
    for r in split_rows:
        if r["split"] in ("val", "test"):
            final_rows.append(r)

    for r in final_rows:
        dest = OUT / r["split"] / r["class"] / r["path"].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(r["path"], dest)
        folder = r["folder"]
        # is_plantvillage is decided by source folder only (exact). The kept ex-`healthy` images
        # come from the mixed field dataset and were visually confirmed to be field shots.
        pv = folder.startswith("Strawberry___")
        ev = "source_folder" if pv else "source_folder_not_pv"
        manifest.append(
            {
                "path": str(dest.relative_to(OUT)).replace("\\", "/"),
                "final_class": r["class"],
                "source_folder": folder,
                "is_plantvillage": pv,
                "pv_evidence": ev,
                "split": r["split"],
                "is_dup_copy": r["dup"],
                "dup_source": r.get("dup_src", ""),
                "dhash": format(r["hash"], "016x"),
            }
        )

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["path", "final_class", "source_folder", "is_plantvillage", "pv_evidence",
                        "split", "is_dup_copy", "dup_source", "dhash"],
        )
        w.writeheader()
        w.writerows(manifest)

    # ---------- report ----------
    # counts come from the manifest, excluding oversample copies, so nothing is double-counted
    n = defaultdict(lambda: [0, 0, 0])  # real images per class per split
    pv = defaultdict(lambda: [0, 0, 0])  # plantvillage images per class per split
    tot = Counter()
    for m in manifest:
        if m["is_dup_copy"]:
            continue
        si = {"train": 0, "val": 1, "test": 2}[m["split"]]
        n[m["final_class"]][si] += 1
        tot[m["split"]] += 1
        if m["is_plantvillage"]:
            pv[m["final_class"]][si] += 1

    print("\n=== SPLIT TABLE (real images only, oversample copies excluded) ===")
    hdr = (
        f"{'class':24s} {'total':>6s} {'train':>6s} {'val':>5s} {'test':>5s}"
        f"  {'PVtr':>6s} {'PVval':>6s} {'PVtest':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cls in sorted(CLASS_SOURCES):
        t, v, te = n[cls]
        pt = pv[cls][0] / t if t else 0
        pv_ = pv[cls][1] / v if v else 0
        pte = pv[cls][2] / te if te else 0
        print(f"{cls:24s} {t+v+te:6d} {t:6d} {v:5d} {te:5d}  {pt*100:6.1f}% {pv_*100:5.1f}% {pte*100:5.1f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL':24s} {sum(tot.values()):6d} {tot['train']:6d} {tot['val']:5d} {tot['test']:5d}")

    print(f"\noversample copies added: {len(added_dups)}")
    for cls, p in added_dups:
        print(f"    {cls:24s} {p.name}")
    if capped:
        print(f"capped (> {CAP_TRAIN} train): {capped}")
    print(f"\ndedupe removed: exact={len(exact_dup)} near={len(near_dup)} cross-class={len(cross_class)}")
    for a, c1, c2, d in (exact_dup[:10] + near_dup[:10]):
        print(f"    dist={d:2d} {c1:22s} dup_of {c2:22s} {a.name}")
    if cross_class:
        print("  cross-class conflicts (near-identical image under two different labels):")
        for a, c1, c2, d in cross_class[:20]:
            print(f"    dist={d:2d} {c1:22s} == {c2:22s}  {a.name}")

    stats = {
        "seed": SEED,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "dedupe": {"kept": len(kept), "exact": len(exact_dup), "near": len(near_dup),
                   "cross_class": len(cross_class), "unreadable": len(hash_errs)},
        "oversample_copies": len(added_dups),
        "capped": capped,
        "split_real": {cls: dict(zip(("train", "val", "test"), n[cls])) for cls in n},
        "plantvillage_frac": {cls: dict(zip(("train", "val", "test"),
                                            [round(pv[cls][i] / n[cls][i], 4) if n[cls][i] else 0 for i in range(3)]))
                              for cls in n},
    }
    (OUT / "prep_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\ndataset -> {OUT}\nmanifest -> {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
