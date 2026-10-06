k-NN baseline for the strawberry 9-class classifier.

Same backbone and same preprocessing as the submitted model (YOLOv5s-cls, 224px, RGB),
but the 9-way softmax head is replaced by a distance-weighted k-NN vote over frozen
penultimate features. Any difference from the submitted model is therefore attributable
to the classifier head, not to the representation or the input pipeline.

k was selected on data/val only. data/test was never used to choose k.

  k_sweep.csv               every k tried, val and test metrics
  knn_k3_val/eval.json      k=3 on val
  knn_k3_test/eval.json     k=3 on test
  features/*.npz            cached features
