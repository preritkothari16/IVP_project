"""Label-quality audit, step 2: score every training image of each class against its rival.

Rival selection:
  * where the shipped confusion matrix shows real confusion, use that rival
  * for the four classes at 100% (where the matrix yields a 0-count "rival"), fall back to the
    biologically adjacent class documented in the earlier analysis:
      blossom_blight <-> gray_mold        (both Botrytis cinerea)
      powdery_mildew_leaf <-> powdery_mildew_fruit  (same pathogen, different organ)
      leaf_scorch <-> angular_leafspot   (both necrotic leaf margins)
      healthy <-> angular_leafspot       (pale/lesion-free foliage is the confusion)

Signature (identical method to the powdery_mildew_fruit vs gray_mold audit):
  lesion mask = bright & desaturated (powder-like) UNION grey-brown (mycelium-like)
  features measured inside that mask: luminance, local roughness (gradient), brown share,
  plus overall lesion coverage.

Discriminative features per pair are chosen automatically as those whose IQRs differ most
between the two classes, so the flagging rule is not hand-tuned per class.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

DS = Path(r"D:\ivp\plant_ai\dataset")
TRAIN = DS / "train"

# class -> rival.  value is (rival, why)
RIVALS = {
    "angular_leafspot":            ("leaf_scorch", "cm: als->leaf_scorch in baseline; both necrotic margins"),
    "anthracnose_fruit_rot":       ("powdery_mildew_fruit", "cm: 4/8 confused with it (50%)"),
    "blossom_blight":             ("gray_mold", "no cm errors; same fungus (Botrytis cinerea)"),
    "gray_mold":                  ("powdery_mildew_fruit", "cm: 5/40 both directions"),
    "healthy":                    ("angular_leafspot", "cm: als->healthy 1; healthy->als 1"),
    "leaf_scorch":                ("angular_leafspot", "no cm errors; both necrotic leaf margins"),
    "leaf_spot":                  ("powdery_mildew_fruit", "cm: 1/48 leaf_spot->pmf"),
    "powdery_mildew_fruit":       ("gray_mold", "cm: 5/12 (42%); already audited, re-verified here"),
    "powdery_mildew_leaf":        ("powdery_mildew_fruit", "no cm errors; same pathogen, fruit vs leaf"),
}

FEATS = ("coverage", "lum", "rough", "brown")


def signature(p):
    im = cv2.imread(str(p))
    if im is None:
        return None
    b, g, r = (im[..., i].astype(np.float32) for i in range(3))
    mx, mn = im.max(axis=2).astype(np.float32), im.min(axis=2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    val = mx / 255.0
    powder = (val > 0.60) & (sat < 0.20)
    mycel = (val > 0.30) & (val < 0.75) & (sat < 0.32) & (r > b)
    lesion = powder | mycel
    if lesion.sum() < 40:
        return None
    gi = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(np.diff(gi, axis=1, append=gi[:, -1:]))
    gy = np.abs(np.diff(gi, axis=0, append=gi[-1:, :]))
    grad = np.hypot(gx, gy)
    frac = float(lesion.mean())
    return {
        "coverage": frac,
        "lum": float(val[lesion].mean()),
        "rough": float(grad[lesion].mean()),
        # RATIO of the lesion that is grey-brown (mycelium-like) rather than bright powder.
        # Must be a ratio: a pixel count would scale with image size and make the IQR meaningless.
        "brown": float(mycel.sum()) / max(float(lesion.sum()), 1.0),
    }


def real_train_files(cls):
    """Real images only - exclude hard-mined *_hardN augmented variants."""
    d = TRAIN / cls
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("*.jpg")) if "_hard" not in p.stem]


cache = {}


def sigs(cls):
    if cls not in cache:
        out = []
        for p in real_train_files(cls):
            s = signature(p)
            if s:
                out.append((p.name, s))
        cache[cls] = out
    return cache[cls]


report = {}
print("=" * 100)
print("STEP 2 - SCORING EVERY TRAIN IMAGE AGAINST ITS RIVAL")
print("=" * 100)

for cls, (rival, why) in RIVALS.items():
    mine = sigs(cls)
    theirs = sigs(rival)
    if not mine or not theirs:
        print(f"\n{cls}: missing data (mine={len(mine)} rival={len(theirs)})")
        continue

    # discriminative features = largest IQR separation between the two classes
    sep = {}
    for f in FEATS:
        a = np.array([s[f] for _, s in mine])
        b = np.array([s[f] for _, s in theirs])
        qa = np.percentile(a, [25, 75])
        qb = np.percentile(b, [25, 75])
        # normalised gap between the two IQRs
        scale = max(qa[1] - qa[0], qb[1] - qb[0], 1e-6)
        sep[f] = abs(qa[0] - qb[0]) / scale
    use = [f for f in FEATS if sep[f] > 0.30] or list(FEATS)
    # always evaluate on at most 3 features to avoid over-flagging on noise
    use = sorted(use, key=lambda f: -sep[f])[:3]

    iqr = {f: tuple(np.percentile([s[f] for _, s in theirs], [25, 75])) for f in FEATS}

    flagged = []
    for name, s in mine:
        if all(iqr[f][0] <= s[f] <= iqr[f][1] for f in use):
            flagged.append((name, s))

    report[cls] = {"rival": rival, "why": why, "n_train": len(mine),
                   "features": use, "iqr": iqr, "flagged": flagged,
                   "sep": {f: round(float(v), 2) for f, v in sep.items()}}

    print(f"\n{cls}  (n_train={len(mine)})  vs  rival: {rival}")
    print(f"  rival chosen because: {why}")
    print(f"  discriminative features used: {use}   (separation scores {report[cls]['sep']})")
    print(f"  rival IQR: " + "  ".join(f"{f}=[{iqr[f][0]:.3f},{iqr[f][1]:.3f}]" for f in use))
    print(f"  FLAGGED: {len(flagged)} of {len(mine)} ({100*len(flagged)/max(1,len(mine)):.1f}%)")
    for name, s in sorted(flagged, key=lambda x: x[1]["brown"], reverse=True)[:60]:
        print(f"    {name:38s} cov={s['coverage']:.3f} lum={s['lum']:.3f} "
              f"rough={s['rough']:6.2f} brown={s['brown']:.2f}")

Path(r"D:\ivp\plant_ai\audit_labels_stage2.json").write_text(json.dumps(
    {c: {"rival": v["rival"], "why": v["why"], "n_train": v["n_train"],
         "features": v["features"], "iqr": {f: list(x) for f, x in v["iqr"].items()},
         "flagged": [[n, s] for n, s in v["flagged"]]} for c, v in report.items()},
    indent=2), encoding="utf-8")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"{'class':24s} {'n_train':>8s} {'rival':24s} {'flagged':>8s} {'rate':>7s}")
print("-" * 74)
tot = tf = 0
for c, v in report.items():
    n = len(v["flagged"])
    tot += v["n_train"]
    tf += n
    print(f"{c:24s} {v['n_train']:8d} {v['rival']:24s} {n:8d} {100*n/max(1,v['n_train']):6.1f}%")
print("-" * 74)
print(f"{'TOTAL':24s} {tot:8d} {'':24s} {tf:8d} {100*tf/max(1,tot):6.1f}%")
print("\nwrote audit_labels_stage2.json")