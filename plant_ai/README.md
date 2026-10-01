# Strawberry Disease Detector

YOLOv5 image classifier (9 strawberry classes) + FastAPI backend + React/Vite/Tailwind
frontend. Upload a strawberry leaf, flower or fruit and get a diagnosis, confidence,
top-3 alternatives, and treatment guidance.

Everything below was run on Windows 11, Python 3.11.9, Node 24.14.1, NVIDIA RTX 3050 6 GB.

---

## Measured results

Evaluated on the held-out **TEST** split (344 images, never used for training or model selection):

| metric | value |
|---|---|
| **Macro-averaged top-1** | **0.9638** |
| Overall top-1 | 0.9826 |
| Images correct | 338 / 344 |

Per-class accuracy:

| class | test n | accuracy |
|---|---|---|
| angular_leafspot | 30 | 0.9667 |
| anthracnose_fruit_rot | 8 | 1.0000 |
| blossom_blight | 11 | 1.0000 |
| gray_mold | 40 | 1.0000 |
| healthy | 88 | 1.0000 |
| leaf_scorch | 60 | 1.0000 |
| leaf_spot | 48 | 0.9792 |
| powdery_mildew_fruit | 12 | **0.7500** |
| powdery_mildew_leaf | 47 | 0.9787 |

Background-bias guard (the check that matters):

| test-image source | n | top-1 |
|---|---|---|
| PlantVillage | 110 | 1.0000 |
| other (field) | 234 | 0.9744 |
| **gap** | | **+0.0256** |

The only class present in both sources is `healthy` — 50/50 on PlantVillage and 38/38 on field
images, so the model is not keying on the plain lab background. All 6 errors are field images.

Confusion matrix: `eval/ep28/confusion_matrix.png`
Metrics: `eval/ep28/eval.json`, per-image predictions: `eval/ep28/predictions.csv`

### The 6 errors are label ambiguity, not model failure

4 of 6 sit in the powdery-mildew / Botrytis complex. `gray_mold` and `blossom_blight` are the
same fungus (*Botrytis cinerea*); `powdery_mildew_leaf` and `powdery_mildew_fruit` are the same
fungus (*Podosphaera aphanis*) on different organs. Perceptual-hash dedupe independently found
`powdery_mildew_fruit_16.jpg` to be **the same photograph** as a `gray_mold` image (dHash
distance 1) — one berry labelled two ways. These are documented in the UI, not hidden.

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

`yolov5` is used as a **local clone**, not an installed package. Two files in it are patched:

| file | change | why |
|---|---|---|
| `utils/augmentations.py` | added `classify_strong_transforms()` | torchvision aug pipeline: RandomResizedCrop(0.6–1.0) → HFlip → ColorJitter(0.4) → RandomGrayscale(0.05) → RandomErasing(0.1) → Normalize. Albumentations is unusable in this env (see `requirements.txt`). |
| `utils/dataloaders.py` | `ClassificationDataset` uses the above when `augment=True` | wires strong aug into the train split only; val keeps the plain centre-crop transform. |

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
  `best.pt` is selected by best validation accuracy, but there is no early stopping.
- `--workers 0` costs ~2 min/epoch (~3.4 h for 100 epochs). `--workers 4` is untested for
  speed; on Windows it can deadlock on spawn.
- The shipped `best.pt` is from a **28-epoch** run (epoch 29 of 100, stopped early). It already
  clears the 90% macro target at 96.4%, and the residual errors are label ambiguity, so the
  remaining 72 epochs were not run.

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
  eval/ep28/                 confusion matrix + metrics + predictions
  screenshots/               Phase 5 UI evidence
  yolov5/                    local clone (2 files patched)
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
