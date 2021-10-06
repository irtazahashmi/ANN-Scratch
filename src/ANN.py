import math
import numpy as np
from numpy import genfromtxt

from src.Layer import Layer
from sklearn.metrics import confusion_matrix


class ANN:
    layers: list
    features: np.ndarray
    expect: np.ndarray

    def __init__(self, inp: np.ndarray, expect: np.ndarray, n_classes: int, n_layers: int, n_neurons: list,
                 activation: list):
        assert len(n_neurons) == n_layers, "Number of layers and list of neurons are not equal: " + str(
            len(n_neurons)) + " != " + str(n_layers)
        assert len(activation) == n_layers, "Number of layers and list of activations are not equal: " + str(
            len(activation)) + " != " + str(n_layers)

        self.input = inp
        self.expect = self.encode_labels(expect, n_classes)
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.layers = []

        prev = None
        # generate layers
        for i in range(0, n_layers):
            d = Layer()
            shape = inp.shape[1] if i == 0 else n_neurons[i - 1]
            # generate random weights
            rand_dist = 2.4 / n_neurons[i]
            d.weights = np.random.uniform(-rand_dist, rand_dist, (shape, n_neurons[i]))

            d.activation = activation[i]
            d.inp = prev
            d.output = None

            prev = d
            self.layers.append(d)

        self.l_rate = 0.0001  # learning rate
        self.t = 0.01  # threshold
        self.output = np.zeros(expect.shape)

    # get matrix from labels
    @staticmethod
    def encode_labels(label, n):
        z = np.zeros((label.shape[0], n))
        for i in range(z.shape[0]):
            z[i, int(label[i] - 1)] = 1
        return z

    # get labels from matrix
    @staticmethod
    def encode_output(output):
        z = np.zeros((output.shape[0], 1))
        for i in range(z.shape[0]):
            z[i] = np.argmax(output[i]) + 1
        return z

    def back_propagation(self):
        # initialize error for output layer
        error = np.subtract(self.expect, self.output)

        i: Layer
        for i in reversed(self.layers):
            derivative_activation_fun = i.activation_str_to_fun()[1]  # returns (f(x), f'(x)), [1] is f'(x)
            derivative_activation = derivative_activation_fun(i.output)
            gradient = np.multiply(derivative_activation, error)  # f'(X) * Ek(p)

            inp = i.inp.output if i.inp is not None else self.input  # get input for layer

            delta_w = np.dot(inp.T, gradient) * self.l_rate  # Yj(p) * G(Yk) * alpha􏱕􏱕􏱕􏱕
            i.weights = i.weights + delta_w

            error = gradient.dot(i.weights.T)  # sum of the gradient * error for the next layer

    def feed_forward(self, inp_data=None):
        if inp_data is None:
            inp_data = self.input

        i: Layer
        for i in self.layers:
            inp = i.inp.output if i.inp is not None else inp_data  # get input for the layer
            y = np.dot(inp, i.weights)  # Xi(p) * Wij(p)
            y = np.subtract(y, self.t)  # - threshold
            activation = i.activation_str_to_fun()  # get the activation function
            self.output = i.output = activation[0](y)  # activation output assign it to layer output and general output

    # cross entropy loss
    def loss(self):
        log = self.expect * np.log(self.output)
        return -np.sum(log) / self.output.shape[0]

    # predict the classes of inp
    def predict(self, inp):
        self.feed_forward(inp)
        z = self.encode_output(self.output)
        return z


if __name__ == "__main__":
    np.random.seed(42)
    train_set = 0.2
    test_set = 0.1

    features = genfromtxt('./../data/features.txt', delimiter=',')
    expect_raw = genfromtxt('./../data/targets.txt', delimiter=',')
    expect_reshape = np.reshape(expect_raw, (1, expect_raw.shape[0])).T

    all_data = np.append(features, expect_reshape, axis=1)
    np.random.shuffle(all_data)

    # split data into training, validation, and test sets
    train_data = np.split(all_data, [math.floor(all_data.shape[0] * train_set)])

    t_features = train_data[1][:, :-1]
    t_expect = train_data[1][:, -1]
    e_features = train_data[0][:, :-1]
    e_expect = train_data[0][:, -1]

    t_expect_reshape = np.reshape(t_expect, (1, t_expect.shape[0])).T
    all_data = np.append(t_features, t_expect_reshape, axis=1)
    train_data = np.split(all_data, [math.floor(all_data.shape[0] * test_set)])
    t_features = train_data[1][:, :-1]
    t_expect = train_data[1][:, -1]
    test_features = train_data[0][:, :-1]
    test_expect = train_data[0][:, -1]

    ann = ANN(
        inp=t_features,
        expect=t_expect,
        n_classes=7,
        n_layers=2,
        n_neurons=[30, 7],
        activation=[Layer.Activations.Relu, Layer.Activations.Sigmoid]
    )

    # training loop
    for e in range(0, 200):
        ann.feed_forward()
        # get cross entropy error and mean square
        loss = ann.loss()
        mean_square = np.mean(np.square(ann.expect - ann.output))

        ann.back_propagation()

        predict = ann.predict(e_features)
        test = predict - np.reshape(e_expect, (1, e_expect.shape[0])).T

        predict = ann.predict(t_features)
        test2 = predict - np.reshape(t_expect, (1, t_expect.shape[0])).T

        accuracy = np.count_nonzero(test) / test.shape[0]
        accuracy2 = np.count_nonzero(test2) / test2.shape[0]
        print(str(e + 1) +
              ", Loss mean-square: " + str(mean_square) +
              ", Acc val: " + str(accuracy) +
              ", Acc train: " + str(accuracy2)
              )

    predict = ann.predict(test_features)
    test = predict - np.reshape(test_expect, (1, test_expect.shape[0])).T
    test_data = np.append(np.reshape(test_expect, (1, test_expect.shape[0])).T, predict, axis=1)

    print("Predict test wrong: " + str(np.count_nonzero(test) / test.shape[0] * 100) + "%")
    print(confusion_matrix(test_expect, predict, normalize="true"))

    # generate classes
    features = genfromtxt('./../data/unknown.txt', delimiter=',')
    predict = ann.predict(features)
    np.savetxt("./../data/Group_40_classes.txt", predict.T, delimiter=",", fmt="%1.0f")
