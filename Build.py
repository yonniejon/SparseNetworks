import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import heapq

class Model:

    def fine_tune(self, model, train_data, train_labels, validation_data=None, validation_labels=None,
              hyper_param_search_method=''):
        if validation_data is not None and validation_labels is not None:
            x_train, x_validation, y_train, y_validation = (train_data, validation_data, train_labels, validation_labels)
        else:
            x_train, x_validation, y_train, y_validation = train_test_split(
                train_images, train_labels, stratify=None, test_size=0.1, random_state=0)

        model.fit(x_train, y_train, validation_data=[x_validation, y_validation],
                  epochs=20, callbacks=[EarlyStopping(patience=20)])

    def train(self, model, train_data, train_labels, validation_data=None, validation_labels=None):

        if validation_data is not None and validation_labels is not None:
            x_train, x_validation, y_train, y_validation = (train_data, validation_data, train_labels, validation_labels)
        else:
            x_train, x_validation, y_train, y_validation = train_test_split(
                train_images, train_labels, stratify=None, test_size=0.1, random_state=0)

        model.fit(x_train, y_train, validation_data=[x_validation, y_validation],
                  epochs=10, callbacks=[EarlyStopping(patience=5)])
        return model

class Point:
    def __init__(self, i, j, entry_value):
        self.i = i
        self.j = j
        self.value = entry_value

    def __lt__(self, other):
        if not isinstance(other, Point):
            raise TypeError(f"Argument save must be of type Point, not {type(other)}")

        return abs(self.value) < abs(other.value)

class Column:
    def __init__(self, i, entry_value):
        self.index = i
        self.value = entry_value

    def __lt__(self, other):
        if not isinstance(other, Column):
            raise TypeError(f"Argument save must be of type Column, not {type(other)}")

        return abs(self.value) < abs(other.value)

class UnitPruning:
    def compute_rank(self, weights_matrix):
        ordered_cols = []
        n_rows, n_cols = np.shape(weights_matrix)
        for j in range(0, n_cols):
            col_sum = 0
            for i in range(0, n_rows):
                col_sum = col_sum + (weights_matrix[i][j] ** 2)

            l2_norm = col_sum ** (1/2)
            cur_col = Column(j, l2_norm)
            heapq.heappush(ordered_cols, cur_col)
        return ordered_cols

    def prune_matrix(self, weights_matrix, k):
        """
               Parameters
               ----------
               weights_matrix : np.array
                   The weights matrix of some layer in the network
               k : float in [0, 1]
                   represents the percentage of weights to remove
        """
        ordered_cols = self.compute_rank(weights_matrix)
        n_rows, n_cols = np.shape(weights_matrix)
        to_remove = int(k * n_cols)
        new_weights = weights_matrix
        for j in range(0, to_remove):
            cur_col = ordered_cols[j]
            for i in range(0, n_rows):
                new_weights[i][cur_col.index] = 0.0
        return new_weights

    def prune(self, model, k):
        """
               Parameters
               ----------
               model : keras.Model
                   The model from which to prune weights
               k : float in [0, 1]
                   represents the percentage of weights to remove
        """
        for idx, layer in enumerate(model.layers):
            if (idx != 0) and (idx != len(model.layers)-1):
                layer_weights = layer.get_weights()
                weights_matrix = layer_weights[0] #take weight matrix and not biases
                new_weights = self.prune_matrix(weights_matrix, k)
                layer_weights[0] = new_weights
                layer.set_weights(layer_weights)
        return model

    def prune_and_fine_tune(self, model, k, num_rounds, train_data, train_labels, validation_data=None, validation_labels=None):
        model_trainer = Model()
        model = weight_pruner.prune(model, k)
        model = model_trainer.train(model, train_data, train_labels, validation_data, validation_labels)
        model = weight_pruner.prune(model, k)

        return model

class WeightPruning:
    def compute_rank(self, weights_matrix):
        ordered_indices = []
        n_rows, n_cols = np.shape(weights_matrix)
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                cur_point = Point(i, j, weights_matrix[i][j])
                heapq.heappush(ordered_indices, cur_point)
        return ordered_indices

    def prune_matrix(self, weights_matrix, k):
        """
               Parameters
               ----------
               weights_matrix : np.array
                   The weights matrix of some layer in the network
               k : float in [0, 1]
                   represents the percentage of weights to remove
        """
        ordered_weights = self.compute_rank(weights_matrix)
        n_rows, n_cols = np.shape(weights_matrix)
        to_remove = int(k * n_rows * n_cols)
        new_weights = weights_matrix
        for i in range(0, to_remove):
            cur_point = ordered_weights[i]
            new_weights[cur_point.i][cur_point.j] = 0.0
        return new_weights

    def prune(self, model, k):
        """
               Parameters
               ----------
               model : keras.Model
                   The model from which to prune weights
               k : float in [0, 1]
                   represents the percentage of weights to remove
        """
        for idx, layer in enumerate(model.layers):
            if (idx != 0) and (idx != len(model.layers)-1):
                layer_weights = layer.get_weights()
                weights_matrix = layer_weights[0] #take weight matrix and not biases
                new_weights = self.prune_matrix(weights_matrix, k)
                layer_weights[0] = new_weights
                layer.set_weights(layer_weights)
        return model

    def prune_and_fine_tune(self, model, k, num_rounds, train_data, train_labels, validation_data=None, validation_labels=None):
        model_trainer = Model()
        model = weight_pruner.prune(model, k)
        model = model_trainer.train(model, train_data, train_labels, validation_data, validation_labels)
        model = weight_pruner.prune(model, k)

        return model


def read_data():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def check_percentage_missing(model, tolerance):
    for idx, layer in enumerate(model.layers):
        if (idx != 0) and (idx != len(model.layers) - 1):
            num_non_zeros = 0.0
            layer_weights = layer.get_weights()
            weights_matrix = layer_weights[0]
            n_rows, n_cols = np.shape(weights_matrix)
            for i in range(0, n_rows):
                for j in range(0, n_cols):
                    if (weights_matrix[i][j] == 0.0):
                        num_non_zeros = num_non_zeros + 1
            zero_ratio = num_non_zeros / (float(n_rows) * float(n_cols))
            assert (abs(zero_ratio - el) < tolerance)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = read_data()
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    x_train, x_validation, y_train, y_validation = train_test_split(
        train_images, train_labels, stratify=None, test_size=0.1, random_state=0)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data= [x_validation, y_validation],
              epochs=100, callbacks=[EarlyStopping(patience=20)])
    test_loss, test_acc1 = model.evaluate(test_images, test_labels)
    print('\nTest accuracy:', test_acc1)
    weight_pruner = WeightPruning()

    pruning_amounts = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    test_accuracy_list = []
    test_accuracy_list.append(test_acc1)
    for el in pruning_amounts:
        model = weight_pruner.prune_and_fine_tune(model, el, 1, x_train, y_train, x_validation, y_validation)
        test_loss, test_acc2 = model.evaluate(test_images, test_labels)
        print('\nTest accuracy:', test_acc2)
        for idx, layer in enumerate(model.layers):
            if (idx != 0) and (idx != len(model.layers)-1):
                num_non_zeros = 0.0
                layer_weights = layer.get_weights()
                weights_matrix = layer_weights[0]
                n_rows, n_cols = np.shape(weights_matrix)
                for i in range(0, n_rows):
                    for j in range(0,n_cols):
                        if (weights_matrix[i][j] == 0.0):
                            num_non_zeros = num_non_zeros + 1
                zero_ratio = num_non_zeros / (float(n_rows) * float(n_cols))
                print("el is " + str(el))
                print(str(abs(zero_ratio - el)))
                assert(abs(zero_ratio - el) < 0.04)
        test_accuracy_list.append(test_acc2)

    print(test_accuracy_list)
    pruning_amounts = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]