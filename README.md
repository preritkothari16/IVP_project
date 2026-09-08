# IVP_project

Toolbox-free MATLAB image preprocessing and feature extraction pipeline.

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
