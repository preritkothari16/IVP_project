"""Fast, decisive leakage measurement - NO RETRAINING.

The shipped checkpoint (strawberry9val/weights/best.pt) was trained on the OLD train split.
The new dataset_v2 split guarantees no near-duplicate cluster straddles train/val/test *within
dataset_v2*, but the shipped model did not train on dataset_v2/train - it trained on
dataset/train. So the question that matters is:

    does the NEW test set contain images with a perceptual near-duplicate in the set the
    shipped model actually trained on?

Scoring the shipped model on the new test set therefore yields a generalisation estimate whose
contamination is measurable rather than assumed. Compare against the same model on the old test
set and the difference IS the leakage effect, for the price of two inference passes.

Writes dataset_v2/manifest.csv (test rows only) so evaluate.py can run against dataset_v2.
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(r"D:\ivp\plant_ai")
DS = PROJECT / "dataset"
V2 = PROJECT / "dataset_v2"

rows = list(csv.DictReader((DS / "manifest.csv").open(encoding="utf-8")))
assign = json.loads((PROJECT / "resplit_assign.json").read_text(encoding="utf-8"))
clusters = json.loads((PROJECT / "resplit_clusters.json").read_text(encoding="utf-8"))["clusters"]

by_name = {Path(r["path"]).name: r for r in rows}
old_train_names = {Path(r["path"]).name for r in rows if r["split"] == "train" and r["is_dup_copy"] != "True"}

# --- contamination of the new TEST set w.r.t. the OLD TRAINING set -------------------
new_test_names = {Path(p).name for p in assign if assign[p] == "test"}
old_train_names = {Path(r["path"]).name for r in rows
                   if r["split"] == "train" and r["is_dup_copy"] != "True"}

# A new-test image is contaminated if its near-duplicate cluster also contains a real image
# that the SHIPPED model trained on (i.e. an OLD train image).
contaminated = []
for members in clusters.values():
    names = {Path(m).name for m in members}
    nt = names & new_test_names
    if not nt:
        continue
    if names & old_train_names:
        contaminated.extend(sorted(nt))

contaminated = sorted(set(contaminated))

print("=" * 78)
print("CONTAMINATION OF THE NEW TEST SET RELATIVE TO THE OLD TRAINING SET")
print("=" * 78)
print(f"  new test images                       : {len(new_test_names)}")
print(f"  with a near-dup in OLD train          : {len(contaminated)}"
      f"  ({100*len(contaminated)/len(new_test_names):.1f}%)")
print(f"  clean of near-dup contamination       : {len(new_test_names)-len(contaminated)}")
print()
print("  For contrast, the OLD test set vs the OLD train set it was split alongside:")
old_test_names = [Path(r["path"]).name for r in rows
                  if r["split"] == "test" and r["is_dup_copy"] != "True"]
c2 = 0
for members in clusters.values():
    names = {Path(m).name for m in members}
    if names & set(old_test_names) and names & old_train_names:
        c2 += len(names & set(old_test_names))
print(f"  with a near-dup in OLD train          : {c2}  ({100*c2/len(old_test_names):.1f}%)")

# --- write a test-only manifest for dataset_v2 so evaluate.py can run ---------------
man = V2 / "manifest.csv"
fields = ["path", "final_class", "source_folder", "is_plantvillage", "pv_evidence",
          "split", "is_dup_copy", "dup_source", "dhash"]
out = []
for p in new_test_names:
    r = by_name[p]
    nr = dict(r)
    nr["path"] = f"test/{r['final_class']}/{p}"
    nr["split"] = "test"
    out.append(nr)
with man.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(out)
print(f"\nwrote {man} with {len(out)} test rows")

# also a strictly-clean subset, excluding anything with a near-dup in old train.
# Match on FILENAME: `contaminated` holds OLD-split paths (e.g. train/<class>/<name>) while
# `out` holds new-split paths (test/<class>/<name>), so comparing the strings silently fails.
clean_dir = V2 / "test_clean"
if clean_dir.exists():
    import shutil
    shutil.rmtree(clean_dir)
clean_dir.mkdir(exist_ok=True)
cset = {Path(p).name for p in contaminated}
kept = 0
for r in out:
    name = Path(r["path"]).name
    if name in cset:
        continue
    cls = r["final_class"]
    d = clean_dir / cls
    d.mkdir(parents=True, exist_ok=True)
    src = V2 / "test" / cls / name
    tgt = d / name
    if not tgt.exists():
        try:
            os.link(src, tgt)
        except OSError:
            import shutil
            shutil.copy2(src, tgt)
    kept += 1
clean_man = V2 / "manifest_clean.csv"
with clean_man.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in out:
        if Path(r["path"]).name in cset:
            continue
        nr = dict(r)
        nr["path"] = f"test_clean/{r['final_class']}/{Path(r['path']).name}"
        w.writerow(nr)
print(f"wrote test_clean/ with {kept} images and {clean_man.name}")
assert kept == len(new_test_names) - len(contaminated), (
    f"clean subset {kept} != expected {len(new_test_names)-len(contaminated)}")

Path(PROJECT / "leakage_report.json").write_text(json.dumps({
    "new_test_n": len(new_test_names),
    "new_test_contaminated_vs_old_train": len(contaminated),
    "old_test_n": len(old_test_names),
    "old_test_contaminated_vs_old_train": c2,
    "contaminated_paths": contaminated,
}, indent=1), encoding="utf-8")
print("wrote leakage_report.json")