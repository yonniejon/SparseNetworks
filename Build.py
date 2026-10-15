"""
Magnitude-based pruning of a dense Fashion-MNIST classifier.

Trains a baseline network, then for each sparsity level k prunes every hidden
Dense layer (the output layer is never pruned), optionally fine-tunes, and
reports test accuracy.

    python Build.py                        # weight pruning (default)
    python Build.py --method unit          # unit (neuron) pruning
    python Build.py --fine-tune-epochs 0   # prune only, no fine-tuning
    python Build.py --no-mask              # let pruned weights regrow while fine-tuning, then re-prune
    python Build.py --independent          # prune each k from the baseline, not from the previous k
"""
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

keras = tf.keras

# Hidden layer sizes. The first layer is 100 units, as in the original code;
# change it here if your spec calls for something else (e.g. 1000).
HIDDEN_SIZES = [100, 1000, 500, 200]
NUM_CLASSES = 10
PRUNING_AMOUNTS = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #

def prunable_layers(model):
    """Every Dense layer except the final (output) layer."""
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    return dense_layers[:-1]


class Pruner:
    """
    Base class. Subclasses implement `layer_masks`, which returns boolean
    "keep" masks for a layer's kernel and bias (False = pruned).
    """

    def layer_masks(self, kernel, bias, k):
        raise NotImplementedError

    def prune(self, model, k):
        """
        Prune `model` in place.

        Parameters
        ----------
        model : keras.Model
            The model whose hidden Dense layers are pruned.
        k : float in [0, 1]
            Fraction of each layer's weights (or units) to remove.

        Returns
        -------
        list of (layer, kernel_keep, bias_keep)
            The masks that were applied, for use with `MaskWeights`.
        """
        masks = []
        for layer in prunable_layers(model):
            kernel, bias = layer.get_weights()
            kernel_keep, bias_keep = self.layer_masks(kernel, bias, k)
            layer.set_weights([kernel * kernel_keep, bias * bias_keep])
            masks.append((layer, kernel_keep, bias_keep))
        return masks


class WeightPruning(Pruner):
    """Zero the fraction k of individual weights with the smallest |w| in each kernel."""

    def layer_masks(self, kernel, bias, k):
        n_prune = int(k * kernel.size)
        smallest = np.argsort(np.abs(kernel), axis=None, kind="stable")[:n_prune]
        kernel_keep = np.ones(kernel.size, dtype=bool)
        kernel_keep[smallest] = False
        return kernel_keep.reshape(kernel.shape), np.ones(bias.shape, dtype=bool)


class UnitPruning(Pruner):
    """
    Zero the fraction k of units with the smallest L2 norm. A unit is a column
    of the kernel (all weights feeding into one neuron); its bias is zeroed too,
    so the pruned neuron outputs 0 instead of a constant relu(bias).
    """

    def layer_masks(self, kernel, bias, k):
        n_rows, n_cols = kernel.shape
        n_prune = int(k * n_cols)
        weakest = np.argsort(np.linalg.norm(kernel, axis=0), kind="stable")[:n_prune]
        unit_keep = np.ones(n_cols, dtype=bool)
        unit_keep[weakest] = False
        kernel_keep = np.broadcast_to(unit_keep, (n_rows, n_cols)).copy()
        return kernel_keep, unit_keep


class MaskWeights(keras.callbacks.Callback):
    """Re-zero pruned weights after every training batch so they stay pruned while fine-tuning."""

    def __init__(self, masks):
        super().__init__()
        self.pairs = []
        for layer, kernel_keep, bias_keep in masks:
            self.pairs.append((layer.kernel, tf.constant(kernel_keep, dtype=layer.kernel.dtype)))
            self.pairs.append((layer.bias, tf.constant(bias_keep, dtype=layer.bias.dtype)))

    def on_train_batch_end(self, batch, logs=None):
        for variable, keep in self.pairs:
            variable.assign(variable * keep)


def check_sparsity(model, k, tolerance=0.04):
    """Check that each pruned layer's kernel has a fraction of zeros close to k."""
    for layer in prunable_layers(model):
        kernel = layer.get_weights()[0]
        zero_ratio = float(np.mean(kernel == 0.0))
        if abs(zero_ratio - k) >= tolerance:
            raise RuntimeError(f"{layer.name}: {zero_ratio:.3f} of weights are zero, expected about {k}")


# --------------------------------------------------------------------------- #
# Data, model and training
# --------------------------------------------------------------------------- #

def read_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


def compile_model(model):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def build_model():
    model = keras.Sequential(
        [keras.Input(shape=(28, 28)), keras.layers.Flatten()]
        + [keras.layers.Dense(units, activation="relu") for units in HIDDEN_SIZES]
        + [keras.layers.Dense(NUM_CLASSES, activation="softmax")]
    )
    compile_model(model)
    return model


def train(model, train_data, validation_data, epochs, patience, extra_callbacks=()):
    x_train, y_train = train_data
    early_stopping = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[*extra_callbacks, early_stopping],
        verbose=2,
    )
    return model


def prune_and_fine_tune(pruner, model, k, train_data, validation_data, fine_tune_epochs, keep_mask=True):
    masks = pruner.prune(model, k)
    if fine_tune_epochs > 0:
        if keep_mask:
            train(model, train_data, validation_data, fine_tune_epochs, patience=5,
                  extra_callbacks=[MaskWeights(masks)])
        else:
            # Pruned weights grow back during unmasked fine-tuning, so prune again before evaluating.
            train(model, train_data, validation_data, fine_tune_epochs, patience=5)
            pruner.prune(model, k)
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["weight", "unit"], default="weight",
                        help="weight pruning or unit (neuron) pruning (default: weight)")
    parser.add_argument("--epochs", type=int, default=100, help="max epochs for the baseline (default: 100)")
    parser.add_argument("--fine-tune-epochs", type=int, default=10,
                        help="max fine-tuning epochs after each prune; 0 disables fine-tuning (default: 10)")
    parser.add_argument("--no-mask", action="store_true",
                        help="let pruned weights regrow during fine-tuning and prune again afterwards")
    parser.add_argument("--independent", action="store_true",
                        help="prune each level from the trained baseline instead of from the previous level")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    (train_images, train_labels), (test_images, test_labels) = read_data()
    x_train, x_validation, y_train, y_validation = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=args.seed)
    train_data = (x_train, y_train)
    validation_data = (x_validation, y_validation)

    model = build_model()
    train(model, train_data, validation_data, args.epochs, patience=20)
    _, baseline_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nBaseline test accuracy: {baseline_accuracy:.4f}")
    baseline_weights = model.get_weights()

    pruner = UnitPruning() if args.method == "unit" else WeightPruning()
    results = [(0.0, baseline_accuracy)]
    for k in PRUNING_AMOUNTS:
        if args.independent:
            model.set_weights(baseline_weights)
            compile_model(model)  # fresh optimizer state for each level
        prune_and_fine_tune(pruner, model, k, train_data, validation_data,
                            args.fine_tune_epochs, keep_mask=not args.no_mask)
        check_sparsity(model, k)
        _, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"\nk = {k:.2f}  test accuracy: {accuracy:.4f}")
        results.append((k, accuracy))

    print(f"\n{args.method} pruning")
    print("k      test accuracy")
    for k, accuracy in results:
        print(f"{k:<6.2f} {accuracy:.4f}")


if __name__ == "__main__":
    main()
