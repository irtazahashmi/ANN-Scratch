from __future__ import annotations
from enum import Enum

import numpy as np


class Layer(object):
    class Activations(Enum):
        Relu = "relu"
        Sigmoid = "sigmoid"

    weights: np.ndarray
    activation: str
    inp: Layer
    output: np.ndarray

    def __init__(self, weights=None, activation=None, inp=None, output=None):
        self.weights = weights
        self.activation = activation
        self.inp = inp
        self.output = output

    # get activation function and derivative from string
    def activation_str_to_fun(self):
        if self.activation == self.Activations.Relu:
            return Layer.relu, Layer.d_relu
        elif self.activation == self.Activations.Sigmoid:
            return Layer.sigmoid, Layer.d_sigmoid

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        f = Layer.sigmoid(x)
        return f * (1 - f)

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def d_relu(x):
        return 1. * (x > 0)
