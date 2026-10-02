# Strawberry Disease Detector

YOLOv5 image classifier (9 strawberry classes) + FastAPI backend + React/Vite/Tailwind
frontend. Upload a strawberry leaf, flower or fruit and get a diagnosis, confidence,
top-3 alternatives, and treatment guidance.

Everything below was run on Windows 11, Python 3.11.9, Node 24.14.1, NVIDIA RTX 3050 6 GB.

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

### Current final result — class-weighted retrain (2 October 2026)

**Macro-averaged top-1 0.8935 · Overall top-1 0.9651 · 332 / 344 correct (12 errors).**

Shipped checkpoint: `yolov5/runs/train-cls/strawberry9hard/weights/best.pt`, copied to
`backend/best.pt` (SHA-256 `fbd78562…`). Evaluation: `eval/ep100hard/`.

This supersedes the 0.8574 / 0.9506 corrected baseline below, which remains the reference point
for measuring what the retrain bought. The BGR correction history above is unchanged and still
applies to both.

| metric | corrected baseline | **shipped retrain** |
|---|---|---|
| **Macro-averaged top-1** | 0.8574 | **0.8935** |
| Overall top-1 | 0.9506 | **0.9651** |
| Images correct | 327 / 344 | **332 / 344** |
| PV vs field gap | +0.0726 | **+0.0513** |

**Macro accuracy is 89.35%, which is 0.65pp below the original 90% target.** The target is
still missed. `evaluate.py` reports `TARGET 90% macro: BELOW TARGET`.

#### What was changed, and what each change was worth

Three techniques were implemented and measured independently, so the contribution of each is
attributable rather than assumed:

| # | technique | macro | vs. previous |
|---|---|---|---|
| 0 | corrected baseline (BGR bug fixed) | 0.8574 | — |
| 1 | class-weighted CE (4:1 cap) + per-class augmentation | 0.8821 | **+0.0247** |
| 1a | + test-time augmentation, 8 views | 0.8787 | **−0.0034 — rejected** |
| 2 | + hard-example mining | **0.8935** | **+0.0114** |

**1. Class-weighted loss + per-class augmentation (+2.47pp).** `CrossEntropyLoss(weight=…)`
with weights linear in inverse class frequency, capped so the smallest class gets at most a 4:1
boost over the largest, then mean-normalised to 1.0 so the loss scale stayed comparable with the
unweighted run. `healthy` (600 train) received 0.351 against `anthracnose_fruit_rot`'s (58) 1.403.

Augmentation was set per class from measured failure modes, not a blanket setting — see
`class_aug.yaml`. `anthracnose_fruit_rot` got tighter cropping (scale 0.30–0.80) because its errors
are wide shots with a small subject; `powdery_mildew_fruit` got stronger colour jitter because its
errors are discriminative rather than scale-related; `angular_leafspot` got grayscale and erasing
**removed entirely** (5% grayscale and 10% erasing actively destroy the lesion colour and texture
cues that separate it from `leaf_scorch`).

**1a. Test-time augmentation — implemented, measured, and rejected (−0.34pp).** `evaluate.py
--tta 8` averages 8 deterministic views (centre crops at 1.0/0.85/0.7/0.55 × identity/hflip). It
fixed 2 images and **broke 4**, including two `angular_leafspot` images that the retrain had just
fixed. It did lower the low-confidence rate on `anthracnose_fruit_rot` (0.375 → 0.250) but raised
it on `powdery_mildew_fruit` (0.167 → 0.333). Net negative on both headline metrics, so it is
**not** used in the shipped path. The code remains behind a flag for reference.

**2. Hard-example mining (+1.14pp).** For each weak class, 6 TRAIN images were selected whose
measured signature matched that class's *known test failures* — pale low-coverage berries for
`powdery_mildew_fruit`, low subject-fraction frames for `anthracnose_fruit_rot`, low brown-lesion
coverage for `angular_leafspot` — and 4 physically-augmented variants of each were written to the
train split (zoom-out for wider context, tightened crop, desaturated/washed-out, warm-blur). These
are real image transformations of **training** images; no test image was copied or used.

**Rejected on evidence: extended fine-tuning.** The loss curves show convergence, not
undertraining — train loss is flat (−0.0012 across the final 10 epochs) and test loss is *rising*
(+0.0086). The stated precondition for a fine-tuning phase was not met, so it was not run.

#### Full per-class comparison

| class | test n | baseline | Step 1 (weighted+aug) | +TTA8 | **+hard mining** |
|---|---|---|---|---|---|
| angular_leafspot | 30 | 0.8667 | **1.0000** | 0.9333 | 0.9667 |
| anthracnose_fruit_rot | 8 | 0.6250 | 0.5000 | 0.5000 | **0.6250** |
| blossom_blight | 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gray_mold | 40 | 0.9750 | 0.9500 | 0.9750 | 0.9500 |
| healthy | 88 | 1.0000 | 0.9886 | 1.0000 | 1.0000 |
| leaf_scorch | 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| leaf_spot | 48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| powdery_mildew_fruit | 12 | 0.2500 | **0.5000** | 0.5000 | 0.5000 |
| powdery_mildew_leaf | 47 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **MACRO** | | **0.8574** | **0.8821** | **0.8787** | **0.8935** |
| **TOP-1** | | **0.9506** | **0.9622** | **0.9622** | **0.9651** |

Low-confidence rate (fraction of that class's own test images scoring below the 0.6 threshold):

| class | baseline | Step 1 | +TTA8 | **shipped** |
|---|---|---|---|---|
| angular_leafspot | 0.300 | 0.100 | 0.067 | **0.033** |
| anthracnose_fruit_rot | 0.375 | 0.375 | 0.250 | **0.000** |
| blossom_blight | 0.182 | 0.091 | 0.091 | **0.000** |
| gray_mold | 0.125 | 0.025 | 0.075 | **0.050** |
| healthy | 0.000 | 0.023 | 0.034 | **0.011** |
| leaf_scorch | 0.017 | 0.017 | 0.017 | **0.000** |
| leaf_spot | 0.021 | 0.021 | 0.021 | **0.021** |
| powdery_mildew_fruit | 0.250 | 0.167 | 0.333 | **0.083** |
| powdery_mildew_leaf | 0.064 | 0.000 | 0.000 | **0.000** |
| overall | 0.078 | — | — | **0.017** |

Every class improved on the low-confidence measure except `gray_mold`, which rose from 0.025 to
0.050 (one image) as its single error became two. Overall low-confidence rate fell from 7.8% to
1.7%.

#### Zero-regression check on the classes that were already at 100%

`blossom_blight`, `leaf_scorch`, `leaf_spot`, `powdery_mildew_leaf` and `healthy` are all at
**1.0000** in the shipped model, matching or exceeding their baseline. `healthy` dipped to 0.9886
in the intermediate Step 1 run (one image, `healthy_749`, called `powdery_mildew_leaf`) and was
recovered by hard-example mining. No class ended below its baseline.

#### Why the remaining 0.65pp is a data ceiling, not a fixable training defect

Two classes hold the macro average down: `powdery_mildew_fruit` (0.5000) and
`anthracnose_fruit_rot` (0.6250). Both were diagnosed before retraining and both diagnoses point
at data, not optimisation:

- **No leakage or mislabelling remains.** A full **TRAIN-vs-TRAIN cross-class dHash audit** over
  all 2,389 train images (every cross-class pair, not just the suspected classes) returned
  **0 cross-class pairs at dHash ≤5 and none at dHash ≤3**, against a nearest-cross-class-neighbour
  floor of 7–12 bits per class. 23 within-class near-duplicate pairs exist (6 `angular_leafspot`,
  5 `blossom_blight`, 12 `leaf_spot`) and are harmless. Nothing to fix.
- **The classes are genuinely small.** `powdery_mildew_fruit` has 81 real train images and
  `anthracnose_fruit_rot` has 58. `union_dataset` contains 123 and 85 respectively — the split and
  dedupe have already taken everything available. More images cannot be manufactured.
- **The residual errors are biotic crossovers.** All 6 remaining `powdery_mildew_fruit` errors go
  to `gray_mold` (5) or `anthracnose_fruit_rot` (1) — adjacent fruit-rot classes. A
  powdery-mildew-infected berry can be secondarily colonised by *Botrytis*, so the dataset label
  is genuinely arguable for those photos. Perceptual-hash dedupe previously found
  `powdery_mildew_fruit_16.jpg` to be the *same photograph* as a `gray_mold` image (dHash distance
  1).
- **Symptom severity, not scale, is the discriminator that is missing.** The errors are the
  least-typical members of their own class (`powdery_mildew_fruit` test images average 9.4%
  pale-lesion coverage versus 16.4% in train), i.e. early or mild infections that genuinely
  resemble grey mould. The retrain fixed exactly the images that had clear white powder and
  recovered the least-typical ones; what remains needs more examples of *mild* infection.

`angular_leafspot` is the one class whose domain gap was identified and then largely closed:
0.8667 → 0.9667. It is **0% PlantVillage while `leaf_scorch` is 100% PlantVillage** — the concrete
case of the background-sensitivity effect described below, and the one class where widening
appearance range with per-class augmentation made a measured difference.

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

Shipped model: `eval/ep100hard/confusion_matrix.png`, `eval/ep100hard/eval.json`,
`eval/ep100hard/predictions.csv`.
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
selected with knowledge of it. Treat **0.8935 / 0.9651** as a slightly optimistic estimate, not a
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
  augmentation and hard-example mining (`--name strawberry9hard`): **0.8935 macro / 0.9651 top-1**.
  Reproduce with:

  ```powershell
  cd D:\ivp\plant_ai\yolov5
  python classify/train.py --model yolov5s-cls.pt --data ..\dataset --epochs 100 --img 224 `
    --batch-size 32 --name strawberry9hard --workers 4 --device 0 --exist-ok `
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
  eval/ep100hard/            CITED: shipped model - confusion matrix, metrics, predictions
  eval/ep28_corrected/       corrected baseline (pre-retrain), for before/after comparison
  eval/ep100hard_tta8/       the rejected TTA variant of the shipped model (evidence)
  eval/ep28_BUGGY_DO_NOT_CITE/
                            superseded BGR-channel-order output, historical only
  screenshots/               Phase 5 UI evidence
  pest_audit/                pest-scope audit evidence + contact sheets
  class_aug.yaml             per-class strong-augmentation overrides (from failure diagnosis)
  yolov5/                    local clone (4 files patched)
  build_binary_dataset.py, score_healthy.py, prepare_final_dataset.py,
  validate_dataset.py, evaluate.py, start_servers.py, screenshot.py
```

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
- **Macro accuracy is 0.8935, which is 0.65pp below the 90% target** set for this project. The
  originally published 0.9638 met it, but that number came from a channel-order bug (see the
  correction note at the top) and has been withdrawn; the honest figure after fixing the bug was
  0.8574, and the class-weighted retrain recovered to 0.8935.
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
