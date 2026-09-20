# Results

These outputs were generated from the supplied training and test datasets.

- `model_comparison.csv` — performance, runtime, and sparsity summary
- `convergence_comparison.png` — regularized objective by iteration
- `roc_fbs.png` — one-vs-rest ROC curves for FBS
- `roc_subgradient.png` — one-vs-rest ROC curves for subgradient descent
- `confusion_matrix_fbs.png` — FBS confusion matrix
- `confusion_matrix_subgradient.png` — subgradient confusion matrix

Run `python -m src.train` to regenerate them.
