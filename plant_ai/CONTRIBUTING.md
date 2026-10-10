# Contributor guide — read this before changing anything

Written for a human *or* an AI agent picking this repository up cold. Everything here is
load-bearing: each item is something that has already cost time or produced a wrong conclusion.

---

## 1. Current state

| item | value |
|---|---|
| Shipped model | `yolov5/runs/train-cls/strawberry9val/weights/best.pt` (= `backend/best.pt`) |
| Shipped TEST metrics | macro-F1 **0.8834**, macro recall **0.8773**, top-1 **0.9593**, 14 errors, 11 confident-wrong |
| Dataset | `dataset/` — 3,474 images, 9 classes. **Do not rename `train`/`val`/`test`.** |
| Leak-free re-split | `dataset_v2/` — hardlinked to `dataset/`, ~0 disk. Rebuildable, not shipped. |
| Tests | `python main.py check`, `python main.py verify`, `python smoke_fullstack.py` |

Everything is committed on `main`, linear history, no force-pushes.

---

## 2. What changed recently, and why

Read this before assuming a metric is wrong or a dataset is broken.

### 2.1 `classification-mk1` licence — resolved, and it is negative
The largest single source (**56% of train images, sole source of 6 of 9 classes**) is **not
deleted** — it was renamed by the same owner to
`nizier193/strawberry-disease-classification`, live, 8.4 GB. Its Kaggle-declared licence is
`Unknown` with an empty licence URL, so **the owner never granted terms**. Default copyright
applies: **`dataset/` is non-redistributable.**

Do not re-run this investigation or advise hunting for "the original download terms" — there are
none. See README → *Dataset citation*.

### 2.2 Label-quality audit — 427 flagged, 0 mislabeled
The IQR-overlap method was applied to all 9 classes. Every flag was visually inspected and
**every one was a correct label**; the flags were appearance overlap, not error. **0 images were
quarantined. The training set is unchanged.**

If you are asked to "clean the data", read `LABEL_AUDIT.md` first. A second cleanup pass found
nothing, same as the first.

### 2.3 Near-duplicate leakage — measured, turns out negligible
13.1% of the old TEST images had a dHash near-duplicate in TRAIN. A cluster-aware re-split was
built (`dataset_v2/`, 0 clusters straddling). The shipped model was then scored on the 102 new
TEST images it had never seen and that carry no near-duplicate in its training set:

| | old TEST (n=344) | leak-free (n=102) |
|---|---|---|
| top-1 | 0.9593 | 0.9608 |
| macro recall | 0.8773 | 0.8955 |
| confident-wrong | 11 | 2 |

**The leakage did not inflate the reported numbers.** The published metrics stand as-is and
**no retrain was performed**. Do not "fix" the split by retraining; the work is done.

### 2.4 Thread caps — `threadcaps.py`
This machine has 16 cores and ~24 GB RAM. The default per-core BLAS pools exhaust the commit
limit and torch fails to load its CUDA DLLs. **`import threadcaps` before `import torch`** is
already wired into `evaluate.py`, `backend/main.py` and `yolov5/classify/train.py`. Keep that
ordering; it is not cosmetic.

### 2.5 Fullstack smoke test — `smoke_fullstack.py`
Exercises `/api/health`, `/api/classes`, and `/api/predict` through the vite proxy. Currently
**9/9 classes correct**. Add it to any change that touches the backend or the frontend contract.

---

## 3. The API contract (get this wrong and you will chase phantom bugs)

`vite.config.js` proxies **only `/api/*`**, stripping the prefix. So:

* frontend calls `/api/health`, `/api/classes`, `/api/predict`
* backend serves `/health`, `/classes`, `/predict`

`/predict` takes **multipart/form-data** with a single field named **`file`**
(`multipart/form-data; name="file"`), *not* JSON with a base64 string. The response field is
**`top_class`**, not `prediction`.

A JSON base64 POST returns **422**. That is a bad test, not a broken server — this exact mistake
was made while writing the smoke test.

CORS is never exercised in dev because the proxy keeps the browser same-origin.

---

## 4. Environment gotchas (Windows / PowerShell 5.1)

These are real and were each hit during this work:

* `&&` and `||` are **not** statement separators. Use `;` or split commands. The shell here also
  rejects a trailing `&` for backgrounding.
* `Invoke-WebRequest` has **no `-Form`** parameter (that is PS 7+). Use Python for multipart, or
  `curl.exe`.
* `$true` is a read-only automatic variable; do not use it as a loop variable name.
* `npm` must be launched as `npm.cmd` for `Start-Process` to accept it.
* The page file can be too small for torch to load (`WinError 1455`) when many processes are
  resident. `threadcaps.py` prevents this; if you hit it anyway, free memory first.

To start the stack by hand (two terminals, or `main.py serve`):

```powershell
cd D:\ivp\plant_ai
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd D:\ivp\plant_ai\frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Then <http://127.0.0.1:5173>.

---

## 5. Rules for changes here

1. **Never tune on the TEST split.** Select on `val`; evaluate on TEST once, at the end. The
   whole validation split exists for this reason.
2. **Do not rename or reshuffle `dataset/train`, `dataset/val`, `dataset/test`.** The shipped
   model's metrics are tied to the current TEST split.
3. **Measure, don't assume.** Several plausible-sounding "fixes" turned out to have nothing
   behind them (data cleaning, split leakage). Verify before acting.
4. **A 100-epoch retrain on this machine is slow** (~2.6 min/epoch with the default batch 32).
   Prefer `--batch-size 64 --cache ram` and a shorter run if you only need a signal. Budget for
   this before launching anything long.
5. **Keep `disease_info.json` marked `"verified": false`** unless a real agronomist has signed
   off. Do not present model output as verified treatment advice.
6. **Do not commit `dataset/`, `dataset_v2/`, `yolov5/`, or `binary_dataset/`** — all gitignored,
   all large, all regenerable.

---

## 6. Known-open items (do not silently "fix" these)

| item | status |
|---|---|
| macro recall **0.8773** vs the project's 0.90 target | gap is entirely the `anthracnose_fruit_rot` / `powdery_mildew_fruit` / `gray_mold` fruit-rot cluster; a phenotype overlap, not label noise |
| `disease_info.json` unverified | needs an agronomist, or strip the guidance |
| `dataset/` non-redistributable | blocked on upstream licence; nothing to do locally |
| TEST split informed earlier recipe design | residual optimism is real and unquantified; the leakage analysis above bounds the near-duplicate part only |

---

## 7. Where things live

| path | what |
|---|---|
| `paths.py` | every path resolves here; nothing hardcodes an absolute path |
| `main.py` | task runner: `check`, `evaluate`, `serve`, `verify`, `walkthrough`, `all` |
| `threadcaps.py` | BLAS/OMP caps, must precede `import torch` |
| `LABEL_AUDIT.md` | the label-quality audit, all 9 classes |
| `smoke_fullstack.py` | end-to-end API smoke test through the proxy |
| `resplit_*.py` | leak measurement / re-split tooling (measurement only; not used for the shipped model) |
| `eval/<tag>/` | `eval.json`, `predictions.csv`, `confusion_matrix.csv` per run |
| `dataset_v2/` | hardlinked leak-free re-split, gitignored |
