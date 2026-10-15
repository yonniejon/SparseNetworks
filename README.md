# SparseNetworks

Magnitude-based pruning of a dense Fashion-MNIST classifier (TensorFlow 2 / Keras).

`python Build.py` trains a baseline network, prunes it to each sparsity level from 0.25 to 0.99,
fine-tunes after each prune, and prints a table of test accuracy. The output layer is never pruned.

## Options

| Flag | Effect |
|---|---|
| `--method weight` | Default. Zero the individual weights with the smallest magnitude in each layer. |
| `--method unit` | Zero whole units: the kernel columns with the smallest L2 norm, and their biases. |
| `--fine-tune-epochs N` | Maximum fine-tuning epochs after each prune (default 10; `0` disables fine-tuning). Pruned weights are held at zero while fine-tuning. |
| `--no-mask` | Let pruned weights regrow during fine-tuning, then prune again before evaluating. |
| `--independent` | Prune each level from the trained baseline instead of from the previous level. |
| `--epochs N` | Maximum epochs for the baseline (default 100). |
| `--seed N` | Random seed (default 0). |

Hidden layer sizes are set by `HIDDEN_SIZES` at the top of `Build.py`.

## Requirements

`tensorflow`, `numpy`, `scikit-learn`
