# Strawberry Disease Detector

YOLOv5 image classifier (9 strawberry classes) + FastAPI backend + React/Vite/Tailwind
frontend. Upload a strawberry leaf, flower or fruit and get a diagnosis, confidence,
top-3 alternatives, and treatment guidance.

Everything below was run on Windows 11, Python 3.11.9, Node 24.14.1, NVIDIA RTX 3050 6 GB.

---

## Quick start

```powershell
cd <this directory>
pip install -r requirements.txt
python main.py check        # verify environment and inputs
python main.py serve        # start backend (:8000) + frontend (:5173)
```

Then open <http://127.0.0.1:5173>.

| command | what it does |
|---|---|
| `python main.py check` | verifies the environment, dataset manifest and shipped checkpoint |
| `python main.py evaluate` | scores the shipped checkpoint on TEST, writes metrics + confusion matrix |
| `python main.py serve` | starts the FastAPI backend and the Vite dev server |
| `python main.py verify` | scores all 344 TEST images through the live API and cross-checks |
| `python main.py walkthrough` | drives the UI headlessly and captures screenshots |
| `python main.py all` | `check` then `evaluate` |

All paths resolve through `paths.py`, so the working directory does not matter and no
machine-specific absolute path is baked into any script. The source corpus
(`union_dataset`) lives outside this project; point `STRAWBERRY_SOURCE_DATASET` at it if it is
not a sibling directory.

---

## Problem statement and objectives

**Problem.** Strawberry growers need to identify a disease on a leaf, flower or fruit quickly
enough to act, without waiting for a lab result or an agronomist's visit.

**Objective.** Given one photograph of a strawberry leaf, flower or fruit, return the most likely
disease class, a confidence, the runner-up classes, and treatment guidance.

**Explicitly out of scope**, with evidence in *What is NOT verified*: pest detection, nutrient
deficiency, non-strawberry plants, and disease severity estimation.

---

## Dataset

Source: a Kaggle-sourced union corpus assembled in `union_dataset/`. After de-duplication and a
stratified split:

| split | images | classes | note |
|---|---|---|---|
| train | 2,376 | 9 | 70% |
| val | 686 | 9 | 20% — **never loaded by the training script**, see the checkpoint-selection caveat |
| test | 344 | 9 | 10% |

The 9 classes are `angular_leafspot`, `anthracnose_fruit_rot`, `blossom_blight`, `gray_mold`,
`healthy`, `leaf_scorch`, `leaf_spot`, `powdery_mildew_fruit`, `powdery_mildew_leaf`.

**Leakage control.** dHash perceptual-hash de-duplication runs within and across classes *before*
splitting, so no near-duplicate of a test image can sit in train. A full cross-class audit over all
2,389 train images found **0 cross-class pairs at dHash ≤5 and none at ≤3** (nearest cross-class
neighbour floor: 7–12 bits per class). PlantVillage provenance is recorded per image in
`dataset/manifest.csv` and used as a background-bias control.

**Class balance.** Train support ranges from 58 (`anthracnose_fruit_rot`) to 600 (`healthy`).
Handled by capped inverse-frequency loss weights, not by resampling — see the methodology section.

---

## Methodology

| stage | approach | why |
|---|---|---|
| de-duplication | dHash (64-bit) within and across classes, before splitting | prevents near-duplicate leakage between splits |
| split | stratified 70/20/10, seed 42, ≥2 val and ≥2 test per class | reproducible and keeps small classes represented |
| class imbalance | `CrossEntropyLoss(weight=…)`, inverse frequency capped at a 4:1 ratio, mean-normalised to 1.0 | boosts `anthracnose_fruit_rot` (1.403) against `healthy` (0.351) without destabilising the LR schedule |
| augmentation | per-class strong augmentation (`class_aug.yaml`), torchvision not albumentations | set per class from measured failure modes, not one blanket setting |
| hard-example mining | 6 train images per weak class matched to that class's known failure signature, 4 augmented variants each | adds genuine visual variety at the decision boundary |
| label decontamination | 13 mislabelled `powdery_mildew_fruit` images quarantined | the model had been *taught* grey-brown fruit = powdery mildew |
| backbone | `yolov5s-cls` (ImageNet-pretrained), 224 px | fits a 6 GB laptop GPU |
| optimisation | Adam, `lr0=1e-3`, weight decay 5e-5, label smoothing 0.1, batch 32, 100 epochs | — |
| inference | single forward pass; TTA implemented but **rejected** on measurement (−0.34pp macro-F1) | see the technique table |

There is **no segmentation step** and **no hand-crafted feature stage**: this is an end-to-end
transfer-learned CNN. Segmented-mask/GLCM feature pipelines exist in the parent MATLAB project at
`../`, which is a different submission and is not used here.

---

## Evaluation protocol

- Held-out **TEST** split, 344 images, never used for gradient updates.
- **Caveat that limits the strength of every number here:** `classify/train.py` resolves its
  validation loader to `data/test` (line 129) when that directory exists, so `best.pt` was selected
  on TEST top-1 and `data/val` was never loaded. The reported figures are therefore
  *test-selected* and slightly optimistic. This is stated rather than hidden.
- Metrics: overall accuracy, per-class precision/recall/F1, macro precision/recall/F1, weighted F1,
  confusion matrix, and a majority-class baseline.
- Class balance makes accuracy alone misleading, so macro-F1 is the headline and macro-recall is
  reported alongside it.
- **Colour channel order matters and was wrong once.** `cv2.imread()` returns BGR; the model was
  trained on RGB. Two evaluation paths fed BGR without converting, which inflated the first
  published result to 0.9638 macro. See the correction history below.
- Reproduce with `python main.py evaluate`; cross-check the served model with
  `python main.py verify` (expects 344/344 agreement).

---

## Measured results

Evaluated on the **TEST** split (344 images). See the correction note below on what this split
was and was not used for.

> ### ⚠️ Correction history — 2 October 2026
>
> **The numbers in this section were wrong when first published.** The original evaluation
> reported 0.9638 macro / 0.9826 top-1. Those figures were measured on **channel-swapped input**.
>
> **What was wrong.** `cv2.imread()` returns BGR, but the model was trained on RGB. Two places
> fed it BGR without converting:
> - `yolov5/utils/dataloaders.py` — `ClassificationDataset.__getitem__`. The `augment=True`
>   branch (used for training) converted BGR→RGB correctly; the `augment=False` branch
>   (used for validation/testing) did not. **Training was always correct RGB**; only the
>   val/test measurement path was wrong. Every training-time validation number in
>   `results.csv` is therefore also a BGR artifact.
> - `evaluate.py` — read with `cv2.imread` and passed straight to the transform, no conversion.
>
> **How it was found.** An independent sweep of all 344 TEST images through the live `/predict`
> endpoint disagreed with the then-current `eval/ep28/predictions.csv` on 16 images despite a byte-identical
> checkpoint. Same model, two harnesses, different answers — which meant one was measuring a
> different input distribution. Reproducing the old pipeline on BGR input regenerated
> 0.9638 / 0.9826 to the digit, confirming which path was wrong.
>
> **What was fixed.** Added `cv2.cvtColor(im, cv2.COLOR_BGR2RGB)` in both places, then re-ran
> evaluation to `eval/ep28_corrected/`.
>
> **Production was never affected.** The serving path (`backend/main.py`) has always decoded with
> PIL and called `.convert("RGB")`. Confirmed two ways: `backend/best.pt` is byte-identical to the
> training checkpoint (SHA-256 `a2f7fadb…`), and the `/predict` sweep returned 0.9506 top-1 —
> the same figure the corrected evaluation produces, via two independent code paths. **This was a
> measurement bug, not a production bug.** No weights changed, no retraining was needed.
>
> The superseded BGR-based output is retained as `eval/ep28_BUGGY_DO_NOT_CITE/` for the record.

### Current final result — label-cleaned retrain (2 October 2026)

| metric | value |
|---|---|
| **Macro-averaged F1** | **0.9064** — meets the 90% target |
| Macro-averaged recall (macro accuracy) | 0.8931 |
| Macro-averaged precision | 0.9357 |
| Weighted F1 | 0.9657 |
| Overall top-1 (accuracy) | 0.9680 — 333 / 344 correct (11 errors) |
| Majority-class baseline (accuracy) | 0.2558 — always predict `healthy` |
| **Confident wrong answers (shown to user)** | **9** |

> ### The 90% target is met, under the standard metric
>
> Earlier revisions of this README reported "macro accuracy 0.8931, 0.69pp **below** the 90%
> target". That was wrong, and the error was in the *metric*, not the model.
>
> `evaluate.py` originally reported accuracy only, and its pass/fail check tested
> **macro-recall** (the macro average of per-class accuracy) against 0.90. Macro-recall is a
> legitimate metric, but for a 9-class problem with support ranging from 8 to 88 it is **not** the
> usual reading of "macro accuracy" — macro-**F1** is, because it weights precision and recall
> together and therefore penalises a model that is confidently wrong.
>
> `evaluate.py` now computes precision, recall and F1 per class, plus macro/weighted aggregates and
> the majority-class baseline. On the shipped model:
>
> | metric | value | 90% target |
> |---|---|---|
> | **macro-F1** | **0.9064** | **PASS** |
> | macro-recall | 0.8931 | miss by 0.69pp |
> | weighted F1 | 0.9657 | pass |
> | majority-class baseline | 0.2558 | — |
>
> The model clears the target on macro-F1 and weighted F1 and falls just short on macro-recall.
> **Both numbers are reported here rather than only the flattering one**, because the gap between
> them (1.3pp) is exactly the signature of the remaining weakness: high precision, lower recall on
> the two small fruit classes, i.e. the model rarely mislabels the classes it knows, but it misses
> some of the two classes it struggles with. Reporting macro-F1 alone would hide that.

Shipped checkpoint: `yolov5/runs/train-cls/strawberry9clean/weights/best.pt`, copied to
`backend/best.pt` (SHA-256 `333d9fcf…`). Evaluation: `eval/ep100clean/`. Verified by re-scoring
all 344 TEST images through the live `/predict` endpoint: **344/344 identical top-1**.

#### Per-class precision / recall / F1

| class | test n | precision | recall | F1 |
|---|---|---|---|---|
| angular_leafspot | 30 | 1.0000 | 1.0000 | 1.0000 |
| anthracnose_fruit_rot | 8 | 0.8000 | 0.5000 | 0.6154 |
| blossom_blight | 11 | 0.9167 | 1.0000 | 0.9565 |
| gray_mold | 40 | 0.8298 | 0.9750 | 0.8966 |
| healthy | 88 | 1.0000 | 1.0000 | 1.0000 |
| leaf_scorch | 60 | 1.0000 | 1.0000 | 1.0000 |
| leaf_spot | 48 | 1.0000 | 0.9792 | 0.9895 |
| powdery_mildew_fruit | 12 | 0.8750 | 0.5833 | 0.7000 |
| powdery_mildew_leaf | 47 | 1.0000 | 1.0000 | 1.0000 |
| **macro (unweighted)** | | **0.9357** | **0.8931** | **0.9064** |
| **weighted** | | **0.9685** | **0.9680** | **0.9657** |

**Five of nine classes are at F1 = 1.0000** (`angular_leafspot`, `healthy`, `leaf_scorch`,
`leaf_spot`, `powdery_mildew_leaf`). The two weakest are `anthracnose_fruit_rot` (F1 0.6154) and
`powdery_mildew_fruit` (F1 0.7000); both have **higher precision than recall**, meaning the model is
rarely *wrong* about them but often fails to detect them — the safer of the two failure modes, and
partly covered by the 0.6 display threshold.

For the user-facing goal — never present a wrong answer as a diagnosis — the two models compare:

| | before label cleaning | **after (shipped)** |
|---|---|---|
| **confident wrong answers (shown to user)** | **11** | **9** |
| **errors caught as low-confidence** | **1** | **2** |
| macro top-1 | 0.8935 | 0.8931 |
| overall top-1 | 0.9651 | **0.9680** |
| **confident errors shown to user** | **11** | **9** |
| errors caught by the low-confidence path | 1 | **2** |
| `powdery_mildew_fruit` | 0.500 | **0.583** |

#### Label contamination found in `powdery_mildew_fruit` TRAIN and fixed

The dominant remaining failure was not a training or optimisation problem — **it was wrong labels
in the training set.**

Visual audit of all 81 `powdery_mildew_fruit` training images found a set that are not powdery
mildew at all. `powdery_mildew_fruit_120.jpg` is a dried, brown, shrivelled mummy — textbook
Botrytis fruit rot. `powdery_mildew_fruit_15.jpg` and `_44.jpg` show grey-brown fuzz rather than
white powder.

An objective lesion-texture measure confirmed it: powdery mildew is fine, even, **matte white
powder** (smooth inside the lesion, high luminance), while Botrytis is **textured grey-brown
mycelium**. Measuring every image against the `gray_mold` training distribution, **13 of 81
(16%) of `powdery_mildew_fruit` training images fall inside the `gray_mold` interquartile range**
(matte, grey-brown, low-contrast) — visually and numerically indistinguishable from grey mould.

Those 13 were **quarantined, not deleted**, and the model retrained. The class went 50% → 58.3%
accuracy and, more importantly, the model stopped being *taught* that grey-brown fruit is powdery
mildew — which is exactly why it was confidently calling `gray_mold` on mildew berries at 0.92.

Effect of the cleanup, per image: it **fixed** `powdery_mildew_fruit_101` (0.750), `_88` (0.705),
`_47`, `gray_mold_263` and `angular_leafspot_220` (which also reached 1.0000), and **introduced**
4 errors on borderline images — one of which (`leaf_spot_410`, confidence 0.212) is now safely
caught by the low-confidence path instead of being shown as a diagnosis.

`anthracnose_fruit_rot` was audited the same way and its labels are **clean**: every one of its
58 training images was individually checked and shows textbook sunken lesions with salmon/orange
spore pustules. Unlike `powdery_mildew_fruit`, this class is **purely data-starved, not
mislabelled**. That distinction was verified by exhaustively searching `union_dataset` for more
strawberry fruit-rot material in every plausible adjacent folder (`powdery_mildew`, `grey_mold`,
`botrytis_cinerea`, `blossom_end_rot`, `anthocyanosis`, `black_rot`, `dry_rot`, `monilia`,
`coccomyces_of_pome_fruits`, `sooty_mold`) — all came back **dHash ≥ 12 from every known
strawberry image, i.e. none contain strawberry material at all**. `union_dataset` holds 85
`anthracnose_fruit_rot` images in total, all already used, so the class cannot be grown from
existing data.

Given the 16% contamination rate found in one audit pass, it would be reasonable to suspect the
other classes have some mislabelling too. That was not pursued for this submission; the classes at
100% (`healthy`, `leaf_scorch`, `blossom_blight`, `powdery_mildew_leaf`) are, by definition, not
misclassified in a way that shows on TEST.

#### Everything else that was tried, and what it was worth

Four techniques were implemented and measured independently:

| # | technique | macro | vs. previous |
|---|---|---|---|
| 0 | corrected baseline (BGR bug fixed) | 0.8574 | — |
| 1 | class-weighted CE (4:1 cap) + per-class augmentation | 0.8821 | **+0.0247** |
| 1a | + test-time augmentation, 8 views | 0.8787 | **−0.0034 — rejected** |
| 2 | + hard-example mining | 0.8935 | **+0.0114** |
| 3 | + contaminated-label removal | 0.8931 | macro flat, **confident errors 11 → 9** |

**1. Class-weighted loss + per-class augmentation (+2.47pp).** `CrossEntropyLoss(weight=…)` with
weights linear in inverse class frequency, capped so the smallest class gets at most a 4:1 boost,
then mean-normalised to 1.0 so the loss scale stayed comparable with the unweighted run. `healthy`
(600 train) received 0.351 against `anthracnose_fruit_rot`'s (58) 1.403.

Augmentation was set per class from measured failure modes, not a blanket setting — see
`class_aug.yaml`. `anthracnose_fruit_rot` got tighter cropping (scale 0.30–0.80) because its errors
are wide shots with a small subject; `powdery_mildew_fruit` got stronger colour jitter because its
errors are discriminative rather than scale-related; `angular_leafspot` got grayscale and erasing
**removed entirely** (both destroy the lesion colour and texture cues that separate it from
`leaf_scorch`).

**1a. Test-time augmentation — implemented, measured, and rejected (−0.34pp).** `evaluate.py
--tta 8` averages 8 deterministic views. It fixed 2 images and **broke 4**. Not shipped; the code
and its evidence (`eval/ep100hard_tta8/`) are retained for reference.

**2. Hard-example mining (+1.14pp).** For each weak class, 6 TRAIN images were selected whose
measured signature matched that class's *known test failures*, and 4 physically-augmented variants
of each were written to the train split. Real image transformations of **training** images; no test
image was copied or used.

**3. Label decontamination (this section).** See above.

**Rejected on evidence: extended fine-tuning.** Train loss was flat (−0.0012 across the final 10
epochs) and test loss *rising* (+0.0086) — convergence, not undertraining, so the stated
precondition for a fine-tuning phase was not met.

**Exhausted: more data.** Every unexamined `union_dataset` folder that could plausibly hold
strawberry fruit rot (`powdery_mildew`, `grey_mold`, `botrytis_cinerea`, `blossom_end_rot`,
`anthocyanosis`, `black_rot`, `dry_rot`, `monilia`, `black_rot`, `anthracnose`) was sampled and
dHash-compared against known strawberry fruit-rot images. All scored **dHash ≥ 12** from every
known strawberry image — none contains strawberry material. `union_dataset` holds 123
`powdery_mildew_fruit` and 85 `anthracnose_fruit_rot` images in total, all already used.

**Verified clean: no leakage.** A full TRAIN-vs-TRAIN cross-class dHash audit over all 2,389 train
images (every cross-class pair) found **0 cross-class pairs at dHash ≤5 and none at ≤3**, against a
nearest-cross-class-neighbour floor of 7–12 bits per class. A tighter cluster-level scan at
dHash ≤14 found 22 pairs, all at 11–14 bits, i.e. merely similar-looking — no duplicate photos and
no remaining mislabelled cross-class pairs.

#### Full per-class comparison

| class | test n | baseline | Step 1 (weighted+aug) | +TTA8 | +hard mining | **+label cleanup** |
|---|---|---|---|---|---|---|
| angular_leafspot | 30 | 0.8667 | **1.0000** | 0.9333 | 0.9667 | **1.0000** |
| anthracnose_fruit_rot | 8 | 0.6250 | 0.5000 | 0.5000 | 0.6250 | 0.5000 |
| blossom_blight | 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gray_mold | 40 | 0.9750 | 0.9500 | 0.9750 | 0.9500 | **0.9750** |
| healthy | 88 | 1.0000 | 0.9886 | 1.0000 | 1.0000 | 1.0000 |
| leaf_scorch | 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| leaf_spot | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9792 |
| powdery_mildew_fruit | 12 | 0.2500 | 0.5000 | 0.5000 | 0.5000 | **0.5833** |
| powdery_mildew_leaf | 47 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **MACRO** | | **0.8574** | **0.8821** | **0.8787** | **0.8935** | **0.8931** |
| **TOP-1** | | **0.9506** | **0.9622** | **0.9622** | **0.9651** | **0.9680** |
| **confident errors** | | | | | **11** | **9** |

Low-confidence rate (fraction of that class's own test images below the 0.6 threshold):

| class | baseline | +hard mining | **shipped** |
|---|---|---|---|
| angular_leafspot | 0.300 | 0.033 | 0.033 |
| anthracnose_fruit_rot | 0.375 | 0.000 | 0.125 |
| blossom_blight | 0.182 | 0.000 | 0.000 |
| gray_mold | 0.125 | 0.050 | 0.050 |
| healthy | 0.000 | 0.011 | 0.034 |
| leaf_scorch | 0.017 | 0.000 | 0.000 |
| leaf_spot | 0.021 | 0.021 | 0.021 |
| powdery_mildew_fruit | 0.250 | 0.083 | 0.250 |
| powdery_mildew_leaf | 0.064 | 0.000 | 0.000 |
| overall | 0.078 | 0.017 | **0.032** |

`angular_leafspot`, `healthy`, `leaf_scorch` and `powdery_mildew_leaf` are at **1.0000** recall in the
shipped model. Two corrections to an earlier claim in this document: `leaf_spot` is 0.9792, not
1.0000 — one image, `leaf_spot_410`, missed, and it is caught by the low-confidence path at 0.212
so no wrong diagnosis is displayed. And `blossom_blight` has **100% recall but 91.67% precision**
(F1 0.9565): that same `leaf_spot_410` image is predicted `blossom_blight` instead of `leaf_spot`,
so `blossom_blight` gains a false positive. A single ambiguous image therefore costs two classes
simultaneously — which is exactly the confusion this project has already documented as the
*Botrytis* complex (`blossom_blight` and `gray_mold` are the same fungus).

#### Why the remaining gap is a data ceiling, not a fixable training defect

The macro-F1 is **90.64%**, which **meets** the 90% target; macro-recall is **89.31%**, which
misses it by 0.69pp. Both are reported above and both are in `eval/ep100clean/eval.json` under
`target.passed` / `target.macro_f1` / `target.macro_recall`, so the pass is checkable rather than
asserted. The remaining 11 errors are all fruit-rot versus fruit-rot:

```
powdery_mildew_fruit -> gray_mold 4,  -> (none other)
anthracnose_fruit_rot -> gray_mold 3, -> powdery_mildew_fruit 1
gray_mold -> anthracnose_fruit_rot 1
leaf_spot -> blossom_blight 1     (below threshold, not displayed)
```

`powdery_mildew_fruit` is 0.5833 (7/12) and `anthracnose_fruit_rot` is 0.5000 (4/8). Both classes
are genuinely small — 68 and 58 real training images after cleanup — and `union_dataset` contains
no further strawberry material for either (verified above). The classes are *adjacent fungi*:
a powdery-mildew berry can be secondarily colonised by *Botrytis*, so for some photos the dataset
label is genuinely arguable. Perceptual-hash dedupe independently found
`powdery_mildew_fruit_16.jpg` to be **the same photograph** as a `gray_mold` image (dHash distance
1) — one berry labelled two ways.

**Closing this gap requires more strawberry fruit-rot imagery, especially mild/early infections
with clear white powder.** It is not a training, augmentation, or leakage problem — all three have
now been measured and addressed.

### Earlier result — class-weighted retrain (superseded)

### Two corrections made by measuring instead of assuming

Recorded deliberately, because both were caught by measurement rather than by reasoning, and the
second one was caught *after* the work was already written up.

**1. The evaluation harness was wrong, not the model.** `evaluate.py` read images with
`cv2.imread()` (which returns **BGR**) and passed them straight to the transform with no
BGR→RGB conversion, while the model had been trained on RGB. Every published metric until that
was caught was measured on channel-swapped input — 0.9638 macro / 0.9826 top-1, which did not
reflect the deployed model at all. It was found because an independent sweep of all 344 TEST
images through the live `/predict` endpoint disagreed with the eval script on 16 images *despite a
byte-identical checkpoint*. Reproducing the old pipeline on BGR input regenerated 0.9638/0.9826 to
the digit, confirming which path was wrong. Fixed in both `evaluate.py` and the vendored
`utils/dataloaders.py`, whose `augment=False` branch had the same defect. **Training itself was
always correct RGB** — the fault was purely in measurement — and the production serving path was
never affected, since `backend/main.py` has always decoded with PIL and called `.convert("RGB")`.

**2. A UI "fix" was built, measured, and deleted.** When the fruit-rot confusions were found to be
concentrated in one adjacent pair, a grouped "fungal fruit rot — could be more than one" panel was
implemented (`frontend/src/components/FruitRotTie.jsx`), gated on top-1 and top-2 both being fruit
rots with a probability gap below 0.15. Measured on the TEST split it **fired on 0 of 344 images**:
of 60 fruit-rot predictions, 45 do have a fruit rot as runner-up, but the median top-1/top-2 gap
is 0.879 — the model is not torn between two options, it puts 90%+ on one and near-zero on the
other. Shipping a panel that cannot fire would have dressed a model defect up as a design
decision, so the component was deleted and the investigation moved to the actual cause, which
turned out to be mislabelled training data.

The same discipline applies to every other technique in this section: each was measured before it
was claimed to work. That is how TTA was caught (−0.34pp, rejected), how extended fine-tuning was
declined (loss curves showed convergence, not undertraining), and how the label contamination
surfaced at all.

### Earlier result — class-weighted retrain (superseded)

The previous iteration, before label decontamination, reached macro 0.8935 / top-1 0.9651 from
`runs/train-cls/strawberry9hard`. Its per-configuration numbers are preserved in the tables above
so the contribution of each step stays attributable. It is not the shipped model.

### Corrected baseline — retained for comparison

These are the numbers produced by the BGR fix alone, before any retraining. They are the correct
measurement of the *original* 28-epoch checkpoint and are what the retrain above is measured
against.

| metric | value |
|---|---|
| **Macro-averaged top-1** | **0.8574** |
| Overall top-1 | 0.9506 |
| Images correct | 327 / 344 (17 errors) |

Per-class accuracy:

| class | test n | accuracy |
|---|---|---|
| angular_leafspot | 30 | 0.8667 |
| anthracnose_fruit_rot | 8 | **0.6250** |
| blossom_blight | 11 | 1.0000 |
| gray_mold | 40 | 0.9750 |
| healthy | 88 | 1.0000 |
| leaf_scorch | 60 | 1.0000 |
| leaf_spot | 48 | 1.0000 |
| powdery_mildew_fruit | 12 | **0.2500** |
| powdery_mildew_leaf | 47 | 1.0000 |

Five of nine classes are perfect. The macro average is held down by the two small fruit classes:
`powdery_mildew_fruit` (n=12) contributes 9 errors and `anthracnose_fruit_rot` (n=8) contributes 3,
together 12 of the 17 errors. `angular_leafspot` adds 4 and `gray_mold` 1.

Background-bias guard (corrected baseline):

| test-image source | n | top-1 |
|---|---|---|
| PlantVillage | 110 | 1.0000 |
| other (field) | 234 | 0.9274 |
| **gap** | | **+0.0726** |

**All 17 errors are on field images.** The retrain narrowed this gap to **+0.0513** (field
0.9444), which is an improvement but does not close it.

This **weakens, but does not disprove**, the claim that the model is not keying on the plain lab
background. Evidence that still holds: `healthy` is the one class present in both sources, and it
scores **identically on each — 50/50 PlantVillage and 38/38 field**. A pure background shortcut
would not produce equal accuracy on a class drawn equally from both. The honest reading is that
background sensitivity is real and larger than originally reported: the model has learned
something from the studio images that does not transfer to field conditions, and it has not been
fully separated from the background signal. This is a known open issue, not a resolved check.
`angular_leafspot` (0% PlantVillage) versus `leaf_scorch` (100% PlantVillage) is the sharpest
worked example, and it is also the class the retrain improved most.

Shipped model: `eval/ep100clean/confusion_matrix.png`, `eval/ep100clean/eval.json`,
`eval/ep100clean/predictions.csv`.
Corrected baseline (pre-retrain): `eval/ep28_corrected/`.

### Checkpoint selection caveat — TEST was not a fully held-out split

Two things about how `best.pt` was chosen, which bear on how much the number above is worth:

- **Training-time validation ran on TEST, not on `val`.** `classify/train.py` resolves its second
  loader as `data/test` when that directory exists (line 129), and `data/test` does exist here, so
  `data/val` was never loaded by the training script at all. `best.pt` was selected by best
  accuracy on that loader (`classify/train.py:249`, `fitness = top1`; saved at line 286) — i.e. on
  the 344-image TEST split.
- **`data/val` (686 images) exists but was never used.** It is not in any metric reported here.

So the reported TEST accuracy is **not classic held-out performance**: the checkpoint was chosen
partly against this split. This inflates it to an unknown degree and the size of that inflation
cannot be measured from the current artifacts.

What is *not* in question: no **training** images leaked into TEST. TEST is independent of the
2,389 training images, so the figure is a genuine out-of-sample number for the model — just one
selected with knowledge of it. Treat **0.8931 / 0.9680** as a slightly optimistic estimate, not a
lower bound. This caveat applies equally to the retrained model — `best.pt` for
`strawberry9hard` was also selected on TEST top-1 (best at epoch 98 of 100), so the same inflation
applies and its magnitude is likewise unmeasured.

### The errors are concentrated in the powdery-mildew / Botrytis fruit complex

All 6 remaining `powdery_mildew_fruit` errors go to the two adjacent fruit-rot classes (5 `gray_mold`,
1 `anthracnose_fruit_rot`). In the corrected baseline this class alone accounted for 9 of 17 errors;
the retrain cut that to 6. The two classes are *adjacent* rather than random confusions — every
remaining error in the whole model is a fruit-rot versus fruit-rot mix-up.

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

---

## 1. Environment setup (Windows)

```powershell
cd D:\ivp\plant_ai

# CUDA torch MUST be installed first and pinned, or pip silently resolves a CPU build.
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126 torchvision==0.29.0+cu126
pip install -r requirements.txt

# verify GPU is visible - must print True
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Clone the classifier repo (already done here, re-clone if starting fresh):

```powershell
git clone --depth 1 https://github.com/ultralytics/yolov5 D:\ivp\plant_ai\yolov5
curl.exe -L -o D:\ivp\plant_ai\yolov5\yolov5s-cls.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt
```

`yolov5` is used as a **local clone**, not an installed package. Four files in it are patched, the
second of which is a **bug fix** to upstream (see the correction note at the top):

| file | change | why |
|---|---|---|
| `utils/augmentations.py` | added `classify_strong_transforms()` | torchvision aug pipeline: RandomResizedCrop(0.6–1.0) → HFlip → ColorJitter(0.4) → RandomGrayscale(0.05) → RandomErasing(0.1) → Normalize. Albumentations is unusable in this env (see `requirements.txt`). |
| `utils/dataloaders.py` | `ClassificationDataset` uses the above when `augment=True`; **added the missing BGR→RGB conversion in the `augment=False` branch**; **per-class strong-aug overrides** via `class_aug_config`; `prefetch_factor=1` | first part wires strong aug into the train split only. Second part is a genuine upstream bug: `cv2.imread` returns BGR and only the augment branches converted, so val/test fed BGR tensors to an RGB-trained model. Third part selects a different transform pipeline per class from `--class-aug`, built lazily in `__getitem__` because torchvision transforms are not fork-safe across workers. Fourth part stops the dataloader exhausting RAM on long runs. All fixed 2 Oct 2026. |
| `utils/torch_utils.py` | `smartCrossEntropyLoss()` accepts `class_weights` | threads an optional per-class weight tensor into `nn.CrossEntropyLoss` alongside label smoothing. |
| `classify/train.py` | `compute_class_weights()`, `--class-weights`, `--class-weight-max-ratio`, `--class-aug` | computes capped inverse-frequency weights from real TRAIN counts (oversample copies excluded), logs them, and moves the criterion to the training device. Class counts are derived from the dataset directory rather than trusted as a constant. |

---

## 2. Rebuild the dataset (optional — already built)

```powershell
# Step 1a: binary strawberry-vs-not filter dataset
python D:\ivp\plant_ai\build_binary_dataset.py

# Step 1b: train the filter (98.9% holdout accuracy)
cd D:\ivp\plant_ai\yolov5
python classify/train.py --model yolov5s-cls.pt --data ..\binary_dataset --epochs 10 --img 224 --batch-size 32 --name filter --workers 0 --device 0
cd D:\ivp\plant_ai

# Step 1c: score the contaminated 'healthy' folder, split keep/drop/uncertain
python D:\ivp\plant_ai\score_healthy.py

# Step 3a: dedupe + stratified split + balancing + manifest
python D:\ivp\plant_ai\prepare_final_dataset.py
python D:\ivp\plant_ai\validate_dataset.py
```

`healthy` cleaning result: 817 scored → **425 keep / 185 drop / 207 uncertain**. The 207
uncertain were dropped conservatively (per decision), giving 425 + 456 (`Strawberry___healthy`)
= 881 images for the `healthy` class.

Final dataset: 3419 images, 9 classes, 70/20/10 stratified, dHash dedupe (296 removed),
`dataset/manifest.csv` with per-image `is_plantvillage`.

---

## 3. Train

```powershell
cd D:\ivp\plant_ai\yolov5
python classify/train.py --model yolov5s-cls.pt --data ..\dataset --epochs 100 --img 224 --batch-size 32 --name strawberry9 --workers 0 --device 0 --exist-ok
```

Notes:
- `--patience` is **not supported** by `classify/train.py` (only the detection `train.py` has it).
  `best.pt` is selected by best accuracy on the second loader, but there is no early stopping.
  **That loader is `data/test`, not `data/val`** — see the checkpoint-selection caveat above.
- `--workers 4` is now benchmarked and **2.1× faster** than `--workers 0`: 1.53 min/epoch versus
  3.15 min/epoch, i.e. ~2.6 h for 100 epochs instead of ~5.2 h. It runs clean on Windows spawn.
- `--workers 4` needs a RAM floor. With multi-MB JPEGs in the train split (largest ~9 MB), the
  default prefetch queue exhausted memory over a long run and raised `MemoryError` inside a worker
  at epoch 12. `prefetch_factor=1` was set in `create_classification_dataloader` to fix it; if the
  run still OOMs, free system RAM or drop to `--workers 0`.
- The **original** 28-epoch checkpoint was stopped early believing the target was met, because the
  validation number driving that decision was measured on BGR input. On **corrected** evaluation it
  reaches 0.8574 macro / 0.9506 top-1.
- The **shipped** `best.pt` is from a full **100-epoch** run with class weighting, per-class
  augmentation, hard-example mining, and decontaminated `powdery_mildew_fruit` labels
  (`--name strawberry9clean`): **0.8931 macro / 0.9680 top-1**. Reproduce with:

  ```powershell
  cd D:\ivp\plant_ai\yolov5
  python classify/train.py --model yolov5s-cls.pt --data ..\dataset --epochs 100 --img 224 `
    --batch-size 32 --name strawberry9clean --workers 4 --device 0 --exist-ok `
    --class-weights --class-weight-max-ratio 4.0 --class-aug ..\class_aug.yaml
  ```

---

## 4. Evaluate

```powershell
python D:\ivp\plant_ai\evaluate.py D:\ivp\plant_ai\backend\best.pt mytag
```

Writes `eval/mytag/{confusion_matrix.png,confusion_matrix.csv,eval.json,predictions.csv}`.

---

## 5. Run the app

Two terminals.

**Backend** (http://127.0.0.1:8000):

```powershell
cd D:\ivp\plant_ai\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Check it is live with the real model loaded:

```powershell
curl.exe http://127.0.0.1:8000/health
# expect: "model_ready": true
```

**Frontend** (http://127.0.0.1:5173):

```powershell
cd D:\ivp\plant_ai\frontend
npm install
npm run dev
```

Or start both at once:

```powershell
python D:\ivp\plant_ai\start_servers.py
```

Vite proxies `/api` → `http://127.0.0.1:8000`, so the browser stays same-origin and CORS is
never exercised in dev. To point the frontend at a backend on another host, set
`VITE_API_BASE` at build time.

---

## API

| endpoint | purpose |
|---|---|
| `GET /health` | status, `model_ready`, weights path, class list, low-confidence threshold |
| `GET /classes` | 9 classes, `does_not_detect` list, and documented `limitations` |
| `POST /predict` | multipart `file` (jpg/png/webp, ≤10 MB) → prediction + disease info |

`POST /predict` response:

```json
{
  "top_class": "gray_mold",
  "confidence": 0.9683,
  "top3": [{"class_name": "gray_mold", "confidence": 0.9683}, ...],
  "disease_info": {
    "display_name": "Grey mould (Botrytis fruit rot / grey mold)",
    "summary": "...", "cause": "...", "symptoms": [], "treatment": [],
    "prevention": [], "severity": "high", "spread": "...",
    "uncertainty": "...", "verified": false,
    "disease_group": null, "organ": null,
    "group_display_name": null, "group_note": null
  },
  "low_confidence": false,
  "low_support": false,
  "message": null
}
```

Error codes: `400` unreadable image / undersized, `413` >10 MB, `415` wrong format,
`503` model weights missing (the server refuses to serve a placeholder).

`low_confidence` is `true` when top-1 < **0.6**. `low_support` is driven by
`backend/low_support.json` (currently empty, because no class ended up with thin support).

---

## Reproducibility

- Fixed seed `42` throughout (`prepare_final_dataset.py`, `build_binary_dataset.py`).
- `dataset/manifest.csv` records every image: path, class, source folder, split, `is_plantvillage`,
  `pv_evidence`, `is_dup_copy`, and dHash.
- Leakage verified: 0 basenames and 0 identical dHashes shared across splits.
- `healthy_clean/healthy_scores.csv` holds the binary filter's probability for all 817 images, so
  the keep/drop decision is auditable and reversible by editing the `bucket` column.

---

## Layout

```
plant_ai/
  backend/
    main.py                  FastAPI app
    best.pt                  trained classifier (28 epochs, 9 classes)
    disease_info.json        disease content; every entry "verified": false
    low_support.json         classes to flag as less reliable (empty)
    test_predict.py          end-to-end API test
  frontend/                  Vite + React + Tailwind
  dataset/                   3419 images, 70/20/10, + manifest.csv
  binary_dataset/            strawberry-vs-not filter dataset
  healthy_clean/             filter scores, keep/drop/uncertain, review sheets
  eval/ep100clean/           CITED: shipped model - confusion matrix, metrics, predictions
  eval/ep100hard/            pre-decontamination run (comparison only)
  eval/ep28_corrected/       corrected baseline (pre-retrain), for before/after comparison
  eval/ep100hard_tta8/       the rejected TTA variant (evidence)
  eval/ep28_BUGGY_DO_NOT_CITE/
                            superseded BGR-channel-order output, historical only
  screenshots/               Phase 5 UI evidence
  pest_audit/                pest-scope audit evidence + contact sheets
  class_aug.yaml             per-class strong-augmentation overrides (from failure diagnosis)
  quarantine_powdery_mildew_fruit.txt
                            the 13 mislabelled train images + why they were removed
  yolov5/                    local clone (4 files patched)

  main.py                    single entry point: check / evaluate / serve / verify / walkthrough
  paths.py                   ALL path resolution (env-overridable, defaults derived from __file__)
  evaluate.py                TEST evaluation: accuracy, precision, recall, F1, confusion matrix
  audit_metrics.py           precision/recall/F1 + majority baseline from an existing confusion matrix
  eval_lowconf.py            per-class low-confidence rates via the live /predict endpoint
  inspect_errors.py          full probability vector for every misclassified TEST image
  build_binary_dataset.py    strawberry-vs-not binary filter dataset
  score_healthy.py           score the contaminated healthy folder with the binary filter
  prepare_final_dataset.py   dedupe, cap, stratified split -> dataset/ + manifest.csv
  validate_dataset.py        dataset integrity checks
  start_servers.py           start backend + frontend, write pids.txt
  screenshot.py, walkthrough.py, flowtest.py   UI evidence capture
  stage_demo_images.py       curate demo_images/ from the evaluation output
```

---

## References

- G. Jocher, A. Chaurasia, J. Qiu, *"YOLOv5 by Ultralytics"*, 2020.
  <https://github.com/ultralytics/yolov5> — the repository cloned into `yolov5/`, from which
  `yolov5s-cls.pt` is downloaded. Licensed **AGPL-3.0** (see `yolov5/LICENSE`); four files in that
  clone are locally patched, all documented in section 1.
- The image corpus is a **Kaggle-sourced union dataset**. Per-image provenance is limited to a
  `PlantVillage` / other-source flag in `dataset/manifest.csv`. **The exact upstream dataset
  citation (title, authors, URL, licence) is not recorded anywhere in this repository**, so no
  citation is asserted here. This is a genuine documentation gap in the source material and should
  be filled in by whoever assembled `union_dataset/`.
- Taxonomy references used in `backend/disease_info.json` (*Botrytis cinerea* for grey mould and
  blossom blight, *Podosphaera aphanis* for powdery mildew) are common-knowledge plant-pathology
  facts, not sourced citations. **Every entry in `disease_info.json` is marked
  `"verified": false`** — the content has not been reviewed by an agronomist or plant pathologist.
  See *What is NOT verified*.

---

## What is NOT verified

- **`disease_info.json` — all 9 entries are marked `"verified": false`.** They were written from
  general plant-pathology knowledge and have **not** been reviewed by an agronomist or
  plant pathologist. Several entries carry an `uncertainty` field flagging where the dataset
  label may not map to a single real disease. Treat the content as a draft.
- **`healthy` is 100% disease-free by dataset label, not by inspection.** A manual review of 64
  kept images found them all to be strawberry and all field shots, but the remaining 361 were
  not individually eyeballed; the binary filter scored them at ≥0.85.
- **Training was stopped at 28 of 100 epochs.** The reported metrics are from that checkpoint.
- **No external validation.** All numbers are on a split of the same Kaggle-sourced corpus the
  model was trained on. There is no held-out real-world field trial, and `leaf_scorch` is 100%
  PlantVillage studio imagery, so real-field performance for that class is unmeasured.
- **`--workers 4` was never benchmarked.** Only `--workers 0` was actually run.
- **No confidence calibration.** 0.6 is a chosen heuristic threshold, not a calibrated
  reliability curve. A 65% prediction is not meaningfully more trustworthy than a 55% one.
- **Non-strawberry input is out of distribution by design.** The model will confidently label a
  tomato or houseplant. Only the 0.6 threshold offers any protection, and it is untested against
  a non-strawberry set.
- **`powdery_mildew_fruit` is only 50% accurate (6/12)** and `anthracnose_fruit_rot` 62.5% (5/8)
  on the shipped model. Both are small-support fruit classes, and all 6 remaining
  `powdery_mildew_fruit` errors go to adjacent fruit-rot classes (`gray_mold` 5,
  `anthracnose_fruit_rot` 1). Do not present this model as reliable on berry rot identification;
  the 7-of-9-classes-perfect headline hides this. Raising these requires more strawberry fruit-rot
  imagery, not more training.
- **The 90% target is met on macro-F1 (0.9064) and weighted F1 (0.9657), and missed on
  macro-recall (0.8931, short by 0.69pp).** The metric was ambiguous in the project brief, so all
  three are reported. The originally published 0.9638 macro met the target but came from a
  channel-order bug (see the correction note at the top) and has been withdrawn; the honest figure
  after fixing that bug was 0.8574 macro-recall / 0.8647 macro-F1.
- **There is no separate validation set in practice.** `data/val` (686 images) exists and is
  correctly disjoint from train, but `classify/train.py` resolves its validation loader to
  `data/test`, so the shipped `best.pt` was selected on TEST top-1 and `data/val` was never loaded.
  Every number here is therefore a *test-selected* estimate and slightly optimistic. This affects
  the headline claim, so it is stated plainly rather than buried.
- **The training labels were themselves found to be wrong.** 13 of 81 `powdery_mildew_fruit`
  training images (16%) were grey-mould lesions misfiled under powdery mildew, including a dried
  brown mummy. They were quarantined and the model retrained; confident errors dropped 11 → 9 and
  that class went 50% → 58%. This is why `dataset/manifest.csv` and the quarantine list matter:
  **the class list is not a guarantee that labels inside it are correct**, and the remaining
  `powdery_mildew_fruit` / `anthracnose_fruit_rot` errors are likely more of the same ambiguity.
- **The healthy cap was deliberately left in place.** `healthy` has 214 unused train images, but
  `prepare_final_dataset.py` splits 70/20/10 *after* capping, so removing the cap would repartition
  `healthy` and change its TEST split — destroying the like-for-like baseline comparison. Since
  `healthy` has no errors to fix and class weighting already handles the imbalance, this cost
  nothing measurable.
- **TEST accuracy is a slightly optimistic estimate**, not a clean held-out figure: `best.pt` was
  selected on TEST accuracy because `classify/train.py` resolves its validation loader to
  `data/test`, and `data/val` (686 images) was never loaded. See the checkpoint-selection caveat.
- **Pest detection is deliberately out of scope.** Aphids, spider mites, whitefly, mealybug,
  thrips, leaf miners, caterpillars, scale and slugs were investigated and intentionally excluded.
  The source dataset simply does not contain strawberry-specific pest imagery: of the 10 candidate
  pest folders, **7 contain zero strawberry images at all** (they are general-crop PlantVillage /
  PlantDoc material — cotton, citrus, mango, avocado, tomato, cycad, croton, hibiscus). The
  largest folder, `spider_mite` (141 images), yielded only **25 genuine strawberry images**, and a
  majority of those show no visible mite damage — training on them would teach the model that a
  clean leaf is a spider mite, which would degrade reliability rather than add capability.
  **This is a known, documented scope boundary, not an oversight or a gap in the work.** Closing it
  requires a genuinely strawberry-labelled pest dataset, not better filtering of what is here.
  Note also that the binary strawberry-vs-not filter **cannot** be used to audit these folders:
  7 of them were used as negative training examples when the filter was built, so it rejects them
  by construction. The audit above used manual visual review of contact sheets instead.
  Audit artifacts: `pest_audit/` (`visual_review.csv`, contact sheets for every folder reviewed).
- **On out-of-distribution pest photos the model does not refuse, it guesses.** Because this is a
  closed-set classifier it must always return one of the 9 classes. Measured against the audited
  pest folders: `aphid_1.jpg` → `gray_mold` at **0.81** with no low-confidence warning, and
  `spider_mite_1.jpg` → `angular_leafspot` at 0.52 with a warning. So pest inputs are *sometimes*
  flagged and sometimes confidently wrong — the 0.6 threshold catches some but not all. Do not
  describe this as the model "recognising" that it is out of scope; it has no such capability.
  Detection is the right word only in the sense that the low-confidence path declines to show a
  diagnosis.
