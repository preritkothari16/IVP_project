# Setup, training and the vendored YOLOv5 patch

Detail that does not belong in the top-level README but is needed to reproduce the work or to
understand what was changed in third-party code.

---

## Environment

Developed on Windows 11, Python 3.11.9, Node 24.14.1, NVIDIA RTX 3050 6 GB, CUDA 12.6.

### Install order matters

```bash
# 1. CUDA torch FIRST — otherwise pip silently resolves a CPU build
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126 torchvision==0.29.0+cu126

# 2. the rest
pip install -r requirements.txt

# 3. frontend
cd frontend && npm install && cd ..

# 4. vendored YOLOv5
git clone --depth 1 https://github.com/ultralytics/yolov5 yolov5
```

Verify:

```bash
python main.py check
```

### Thread caps — required on many-core machines

`threadcaps.py` caps `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 2 before
`import torch`, and is wired into `evaluate.py`, `backend/main.py` and
`yolov5/classify/train.py`.

On a 16-core machine with ~24 GB RAM and several hundred resident processes, the default per-core
BLAS pools exhaust the commit limit. Two distinct failures result, and neither looks like what it
is:

```
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

```
OSError: [WinError 1455] The paging file is too small for this operation to complete.
  ... torch/lib/curand64_10.dll
```

The second is easily mistaken for a broken torch installation. Override with
`STRAWBERRY_THREADS` if a different machine needs different values.

### Deliberately not installed

`albumentations` is **not** in `requirements.txt`, for three specific reasons recorded in that
file: the latest release needs `stringzilla` which has no MSVC wheel here; 1.3.x/1.4.x lack
`RandomGrayscale`/`RandomErasing`, which this project uses; and its `qudida` dependency silently
upgrades `opencv-python` to 5.0.0.

Strong augmentation is implemented with torchvision instead, in
`yolov5/utils/augmentations.py :: classify_strong_transforms()`.

---

## The vendored YOLOv5 clone

`yolov5/` is a shallow clone of `ultralytics/yolov5` at `402e17d`, licensed **AGPL-3.0**. Four
files are locally patched:

| file | why |
|---|---|
| `classify/train.py` | validate on `data/val` rather than `data/test`; class weighting; per-class augmentation hooks |
| `utils/dataloaders.py` | weighted sampler driven by real per-class TRAIN counts |
| `utils/augmentations.py` | `classify_strong_transforms()`, torchvision-based |
| `utils/torch_utils.py` | minor compatibility fixes for torch 2.14 |

Check what is currently modified:

```bash
cd yolov5 && git status --short
```

---

## Rebuilding the dataset

Optional — `dataset/` is already built. To rebuild from the source corpus:

```bash
python prepare_final_dataset.py
```

Outputs `dataset/{train,val,test}/<class>/` plus `dataset/manifest.csv`. The manifest records per
image: final class, source folder, PlantVillage flag and the evidence for it, split, augmentation
provenance (`is_dup_copy`, `dup_source`) and a perceptual hash (`dhash`).

> Do not rename or reshuffle `train`, `val` or `test`. The published metrics in the README are tied
> to this exact TEST split.

---

## Training

~2.6 min/epoch at batch 32 on an RTX 3050, so budget roughly 4 hours for 100 epochs. If you only
need a signal, `--batch-size 64 --cache ram` roughly halves that.

```bash
cd yolov5
python classify/train.py --data ../dataset --name strawberry9val --epochs 100 \
  --batch-size 32 --imgsz 224 --class-weights --optimizer Adam --lr0 0.001 --label-smoothing 0.1
```

`class_aug.yaml` holds the per-class augmentation policy — the three small/awkward fruit classes
get gentler jitter and scale ranges than the leaf classes.

### The validation-holdout protocol

This is the part that is easy to get wrong. `classify/train.py` picks its validation directory as
`data/test` if that directory exists, otherwise `data/val`. To validate honestly:

```bash
mv dataset/test dataset/_test_holdout
cd yolov5 && python classify/train.py --data ../dataset ... --name strawberry9val
cd .. && mv dataset/_test_holdout dataset/test
```

`strawberry9val/results.csv` then contains a `val/loss` column rather than `test/loss`. That column
name is the proof the holdout worked — an earlier run selected its checkpoint on TEST and the
measured cost of that mistake was **+0.0231 macro F1**.

---

## Evaluation

```bash
python main.py evaluate                       # shipped checkpoint on TEST
python main.py evaluate --weights path.pt --tag mytag --tta 8
```

Writes `eval/<tag>/eval.json`, `predictions.csv`, `confusion_matrix.csv` and a confusion matrix
PNG.

**`evaluate.py` loads images as RGB.** `cv2.imread` returns BGR and the model was trained on RGB
via torchvision transforms; the missing conversion was a real bug that inflated nothing but
silently mis-scored until fixed. All published numbers are post-fix.

---

## Verifying the live app

```bash
python main.py verify         # score all 344 TEST images through the running API
python smoke_fullstack.py     # end-to-end through the vite proxy, one image per class
```

`verify` is the stronger check: it confirms the deployed `backend/best.pt` agrees with offline
evaluation image-for-image, so a stale or mismatched checkpoint cannot pass silently.

---

## References

- Hughes, D. P. & Salathé, M. *An open access repository of images on plant health.* arXiv:1511.08060 — PlantVillage
- Jocher, G., Chinni, V. & Qi, X. *ultralytics/yolov5*, AGPL-3.0
- Plant pathology taxonomy used in `disease_info.json` (*Botrytis cinerea* for grey mould and
  blossom blight; *Podosphaera aphanis* for powdery mildew) is common knowledge, **not a sourced
  citation**, and every entry is marked `"verified": false`.

Full dataset provenance, including how `leaf_scorch` was identified as PlantVillage and how
`classification-mk1` was traced to its renamed successor, is in the *Data licensing* section of
`README.md`.
