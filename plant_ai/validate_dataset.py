import paths  # central path configuration; see paths.py
import csv
from collections import Counter, defaultdict
from pathlib import Path


OUT = paths.DATASET
rows = list(csv.DictReader((OUT / "manifest.csv").open(encoding="utf-8")))

disk = Counter()
for split in ("train", "val", "test"):
    for d in (OUT / split).iterdir():
        if d.is_dir():
            disk[(split, d.name)] = len([f for f in d.iterdir() if f.is_file()])

print("on-disk images (incl. any oversample copies):")
for k in sorted(disk):
    print(f"  {k[0]:5s} {k[1]:24s} {disk[k]}")
print("  total on disk:", sum(disk.values()))
print("  manifest rows:", len(rows))

# manifest vs disk consistency
mcount = Counter((r["split"], r["final_class"]) for r in rows)
bad = [k for k in set(list(mcount) + list(disk)) if mcount.get(k, 0) != disk.get(k, 0)]
print("  manifest/disk mismatches:", bad if bad else "NONE")

# duplicate-name check across splits (leakage sanity)
names = defaultdict(set)
for r in rows:
    names[r["path"].split("/")[-1]].add(r["split"])
multi = {n: s for n, s in names.items() if len(s) > 1}
print(f"  basenames appearing in >1 split: {len(multi)}")
for n, s in list(multi.items())[:10]:
    print(f"     {n} -> {sorted(s)}")

# dhash overlap across splits
by_hash = defaultdict(set)
for r in rows:
    if r["is_dup_copy"] == "False":
        by_hash[r["dhash"]].add(r["split"])
leaks = {h: s for h, s in by_hash.items() if len(s) > 1}
print(f"  identical dhash across >1 split (must be 0): {len(leaks)}")

# PlantVillage evidence breakdown
ev = Counter((r["source_folder"], r["pv_evidence"]) for r in rows)
print("\n  source_folder -> pv_evidence:")
for k, v in sorted(ev.items()):
    print(f"     {k[0]:26s} {k[1]:22s} {v}")
pv_healthy = [r for r in rows if r["final_class"] == "healthy" and r["source_folder"] == "healthy"]
print(f"\n  healthy from filtered 'healthy' folder: {len(pv_healthy)}, "
      f"flagged PV by heuristic: {sum(1 for r in pv_healthy if r['is_plantvillage']=='True')} "
      f"({100*sum(1 for r in pv_healthy if r['is_plantvillage']=='True')/max(1,len(pv_healthy)):.1f}%)")
