# Measured results and verification detail

The long-form measurement record that used to live inline in `README.md`. Nothing here is
required to run the app; it is the evidence behind every number the README quotes.

Shipped checkpoint: `yolov5/runs/train-cls/strawberry9val/weights/best.pt`, deployed to
`backend/best.pt`. Raw outputs per run live in `eval/<tag>/` as `eval.json`,
`predictions.csv` and `confusion_matrix.csv`.

---

## Authoritative TEST results

`eval/ep100clean_val/eval.json`, 344 held-out images, RGB evaluation (the channel-order bug in
`evaluate.py` is fixed and these numbers are post-fix):

| metric | value |
|---|---|
| top-1 accuracy | **0.9593** |
| macro recall / macro top-1 | **0.8773** |
| macro F1 | **0.8834** |
| macro precision | 0.8915 |
| weighted F1 | 0.9578 |
| majority-class baseline | 0.2558 |
| errors | 14 of 344 |
| confident-wrong (≥ 0.60) | 11 |

### Per class

| class | n | accuracy | F1 |
|---|---|---|---|
| angular_leafspot | 30 | 0.9667 | 0.9831 |
| anthracnose_fruit_rot | 8 | **0.5000** | 0.5333 |
| blossom_blight | 11 | 1.0000 | 1.0000 |
| gray_mold | 40 | 0.9500 | 0.9048 |
| healthy | 88 | 1.0000 | 0.9944 |
| leaf_scorch | 60 | 1.0000 | 1.0000 |
| leaf_spot | 48 | 0.9792 | 0.9895 |
| powdery_mildew_fruit | 12 | **0.5000** | 0.5455 |
| powdery_mildew_leaf | 47 | 1.0000 | 1.0000 |

**Seven of nine classes are at or above 0.95.** 13 of the 14 errors sit inside the three-way
fruit-rot cluster (`anthracnose_fruit_rot` / `powdery_mildew_fruit` / `gray_mold`). The single
error outside it is `angular_leafspot → healthy` at 0.314 confidence. **All 11 confident-wrong
answers (≥ 0.60) are inside the cluster** — none of the other six classes confidently misleads.

### Background-bias guard

| source | n | top-1 |
|---|---|---|
| PlantVillage (studio, 256×256) | 110 | 1.0000 |
| other (field, 419×419) | 234 | 0.9402 |
| gap | | +0.0598 |

The gap is real and is discussed under *Background bias* below.

### The 14 errors

| true | predicted | confidence |
|---|---|---|
| anthracnose_fruit_rot | powdery_mildew_fruit | 0.895 |
| gray_mold | anthracnose_fruit_rot | 0.909 |
| powdery_mildew_fruit | gray_mold | 0.923 |
| powdery_mildew_fruit | anthracnose_fruit_rot | 0.862 |
| anthracnose_fruit_rot | powdery_mildew_fruit | 0.737 |
| powdery_mildew_fruit | gray_mold | 0.714 |
| anthracnose_fruit_rot | gray_mold | 0.872 |
| powdery_mildew_fruit | gray_mold | 0.704 |
| powdery_mildew_fruit | gray_mold | 0.698 |
| powdery_mildew_fruit | gray_mold | 0.696 |
| gray_mold | anthracnose_fruit_rot | 0.350 |
| leaf_spot | powdery_mildew_fruit | 0.286 |
| angular_leafspot | healthy | 0.314 |

### The errors are concentrated in the powdery-mildew / Botrytis fruit complex

Every remaining `powdery_mildew_fruit` error goes to the two adjacent fruit-rot classes. In the
corrected baseline this class alone accounted for 9 of 17 errors; the retrain cut that to 6. The
two classes are *adjacent* rather than random confusions — every remaining error in the whole
model is a fruit-rot versus fruit-rot mix-up.

This is largely genuine label ambiguity in the dataset rather than a model defect: `gray_mold` and
`blossom_blight` are the same fungus (*Botrytis cinerea*); `powdery_mildew_leaf` and
`powdery_mildew_fruit` are the same fungus (*Podosphaera aphanis*) on different organs; and a
powdery-mildew-infected berry can also be colonised by *Botrytis* as a secondary invader, so the
correct label for a given photo is genuinely arguable. Perceptual-hash dedupe independently found
`powdery_mildew_fruit_16.jpg` to be **the same photograph** as a `gray_mold` image (dHash distance
1) — one berry labelled two ways.

That said, "it's the labels" should not be used to excuse the number away. At 50% accuracy on that
class this is a real user-facing limitation, and it is the single thing most worth fixing with
better data or a reconciled labelling pass. It is surfaced in the UI rather than hidden: the app
groups `powdery_mildew_fruit` with the leaf form under one shared explanation, and the
low-confidence path fires on 1 of these 12 images.

Inspecting all 58 `anthracnose_fruit_rot` training images directly confirmed the class is
coherent, textbook anthracnose fruit rot throughout — sunken circular lesions with salmon spore
masses. The test errors are phenotype overlap, not label noise: `anthracnose_fruit_rot_39` shows
a grey-brown fuzzy patch that is arguably real gray mould, `_75` is the bleached over-ripe form
that overlaps fuzzy mould, and `_22` is genuinely ambiguous. One of these may be mislabeled *in
the TEST split*, which is deliberately not touched.

### Background bias

`leaf_scorch` is 100% PlantVillage studio imagery and was the class most easily confused with the
field-shot `angular_leafspot`. Its PlantVillage provenance was confirmed on four independent
signals (256×256 sizing at P=1.00, `Strawberry___<Disease>` folder convention, grey studio
background saturation 0.089 vs 0.24–0.61 for field classes, manifest `is_plantvillage=True`).

### Contamination the source corpus itself flags

51 `leaf_spot` images carry `is_strawberry_disease=False` in the upstream corpus (30 train / 12 val
/ **9 test**). Re-scored with this project's own binary filter they have median `p_strawberry`
0.3596 versus 0.9501 for unflagged controls, and inspection confirms ornamental and
potato/*Rubus*-type foliage. Impact on macro-F1 is −0.0003, but they are *accidentally correct*
out-of-distribution predictions and must not be read as evidence of leaf-spot recognition.

### Checkpoint selection — resolved, and the cost was measured

An earlier run selected its checkpoint on TEST accuracy. The cost of that mistake was measured
rather than assumed: **+0.0231 macro F1** of optimism. The shipped model is selected on
validation. `strawberry9val/results.csv` contains `val/loss`, not `test/loss`, which is the
column-level proof.

### Near-duplicate leakage: measured, and it turns out to be negligible

Clustering all 3,406 real images by dHash (exact pairwise Hamming distance, union-find over
transitive links):

| threshold | near-dup pairs | clusters straddling a split | images in them |
|---|---|---|---|
| ≤ 6 | 36 | 17 | 37 |
| **≤ 10** | **319** | **108** | **318** |
| ≤ 14 | 2,762 | 115 | 1,355 — collapses into one 985-image mega-cluster |

**45 of the 344 TEST images (13.1%) have a perceptual near-duplicate in TRAIN.** Threshold 10 is
used; at 14 transitive chaining collapses everything into one component and the metric stops
meaning anything.

A leak-free group-aware re-split was built (`dataset_v2/`, hardlinked, 69.9/20.1/10.0, **0**
clusters straddling). It was **not** used to retrain. Instead the shipped checkpoint was scored on
the 102 new TEST images it had never seen that carry no near-duplicate in its training set:

| metric | old TEST (n=344, 13.1% leaky) | leak-free (n=102) |
|---|---|---|
| top-1 | 0.9593 | 0.9608 |
| macro recall | 0.8773 | 0.8955 |
| macro-F1 | 0.8834 | 0.8941 |
| confident-wrong | 11 | 2 |
| confident-wrong rate | 3.2% | 2.0% |

**The leakage did not inflate the reported numbers.** Caveat: n=102 is small and `blossom_blight`
has no clean images, so the comparison is indicative, not conclusive.

### Label-quality audit

427 images flagged across all 9 classes by the same IQR method used on
`powdery_mildew_fruit`; **0 confirmed mislabeled, 427 verified correct, 0 quarantined.** Full
record in `LABEL_AUDIT.md`.

### k-NN baseline

Frozen YOLOv5s-cls penultimate features, `k` selected on validation only:

| | macro F1 | top-1 |
|---|---|---|
| k=3, validation | 0.7896 | 0.8717 |
| k=3, TEST | 0.7313 | 0.8372 |

Validation to TEST drops 0.0584 macro F1. Outputs in `baseline_knn/`, sweep in `baseline_knn/k_sweep.csv`.
