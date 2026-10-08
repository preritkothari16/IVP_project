"""Re-split step 1 (corrected): full pairwise near-duplicate clustering.

The earlier bucketing by dhash prefix is unsound: two images within Hamming 10 can differ in
their leading 16 bits and land in different buckets, so pairs were being missed. This does the
exact O(n^2) product with numpy bit popcount, chunked to bound memory.

3406 images -> 5.8M pairs, exact.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DS = Path(r"D:\ivp\plant_ai\dataset")
rows = list(csv.DictReader((DS / "manifest.csv").open(encoding="utf-8")))
real = [r for r in rows if r["is_dup_copy"] != "True" and "_hard" not in r["path"]]
n = len(real)
print(f"real images: {n}   split: {dict(Counter(r['split'] for r in real))}")

H = np.array([int(r["dhash"], 16) for r in real], dtype=np.uint64)

# popcount lookup over uint8 view
_u8 = H.view(np.uint8).reshape(n, 8)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class DSU:
    def __init__(s, n):
        s.p = list(range(n))

    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]
            x = s.p[x]
        return x

    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb:
            s.p[rb] = ra


def clusters_at(thresh):
    dsu = DSU(n)
    npairs = 0
    CH = 512
    for i0 in range(0, n, CH):
        i1 = min(i0 + CH, n)
        xor = np.bitwise_xor(H[i0:i1, None], H[None, :])
        bv = xor.view(np.uint8).reshape(i1 - i0, n, 8)
        dist = POP[bv].sum(axis=2)
        ii, jj = np.where((dist <= thresh) & (np.arange(i0, i1)[:, None] < np.arange(n)[None, :]))
        for a, b in zip(ii.tolist(), jj.tolist()):
            dsu.union(i0 + a, b)
            npairs += 1
    comp = defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)
    return comp, npairs


results = {}
for THRESH in (6, 10, 14, 18):
    comp, npairs = clusters_at(THRESH)
    multi = {k: v for k, v in comp.items() if len(v) > 1}
    straddle, leaked = 0, 0
    per_class = Counter()
    test_imgs_in_straddling = 0
    for idxs in multi.values():
        sp = {real[i]["split"] for i in idxs}
        if len(sp) > 1:
            straddle += 1
            leaked += len(idxs)
            for i in idxs:
                per_class[real[i]["final_class"]] += 1
                if real[i]["split"] == "test":
                    test_imgs_in_straddling += 1
    print()
    print(f"--- exact pairwise, Hamming <= {THRESH} ---")
    print(f"  near-dup pairs       : {npairs}")
    print(f"  multi-image clusters : {len(multi)}   largest {max((len(v) for v in multi.values()), default=1)}")
    print(f"  images in multi      : {sum(len(v) for v in multi.values())}")
    print(f"  >>> clusters straddling a split boundary : {straddle}")
    print(f"  >>> images inside those clusters         : {leaked}  ({100*leaked/n:.2f}% of all)")
    print(f"  >>> of which in TEST                     : {test_imgs_in_straddling}")
    print(f"  by class                                 : {dict(per_class)}")
    results[THRESH] = {"npairs": npairs, "straddle": straddle, "leaked": leaked,
                       "test_in_straddling": test_imgs_in_straddling,
                       "clusters": {str(k): [real[i]["path"] for i in v] for k, v in comp.items()}}

CHOSEN = 10
Path(r"D:\ivp\plant_ai\resplit_clusters.json").write_text(
    json.dumps({"threshold": CHOSEN, "n_images": n,
                "summary": {str(k): {kk: vv for kk, vv in v.items() if kk != "clusters"}
                            for k, v in results.items()},
                "clusters": results[CHOSEN]["clusters"]}, indent=1), encoding="utf-8")
print(f"\nwrote resplit_clusters.json at threshold {CHOSEN}")
print(f"TOTAL exact-pairs vs earlier bucketed estimate at t=10: {results[10]['npairs']} "
      f"(bucketed said 19)")