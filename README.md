# IVP_project

Toolbox-free MATLAB image preprocessing and feature extraction pipeline.

> **This repository contains two independent projects.**
>
> | | |
> |---|---|
> | **repository root** | the MATLAB preprocessing / feature-extraction pipeline documented below |
> | **[`plant_ai/`](plant_ai/)** | a strawberry disease detector — YOLOv5 classifier + FastAPI + React. See [its README](plant_ai/README.md). |
>
> Both are MIT licensed (source code only — see [Licence](#licence)). The two share no code.

## Requirements

The project uses only functions resolved by this MATLAB installation under
`D:\toolbox\matlab\...`. It does not require Image Processing Toolbox or
Statistics and Machine Learning Toolbox functions.

## Run Order

From the MATLAB Command Window:

```matlab
cd D:\ivp
inspect_dataset
visualizePipeline
preprocessDataset
```

`preprocessDataset` asks for the input dataset folder and output folder, then writes
`*_preprocessed.png` images and matching `*_mask.png` files by class.

After preprocessing, run:

```matlab
featureTable = extractFeatures('D:\ivp\preprocessed');
results = evaluateKnn(featureTable, 0.75, 3);
```

`extractFeatures` writes `extractedFeatures.mat` and `extractedFeatures.csv` in the
selected preprocessed folder. `evaluateKnn` performs a reproducible stratified split,
manual k-nearest-neighbor classification, and prints accuracy plus a confusion matrix.

## Licence

MIT — see [LICENSE](LICENSE). The licence is kept as a verbatim licence text so that
GitHub's licence detection classifies it correctly; the caveats live here instead.

**Scope.** MIT covers the **source code** of both projects in this repository. It does
**not** cover, and does not grant any right to redistribute:

- **Image datasets.** The largest upstream source for `plant_ai` declares no licence at
  all, and two further sources are non-commercial with share-alike obligations, so no
  composite licence can be granted. See [Data licensing](plant_ai/README.md#data-licensing--read-before-redistributing).
- **Trained model weights** (`plant_ai/backend/best.pt`).

Third-party components keep their own licences. `ultralytics/yolov5` is AGPL-3.0.
