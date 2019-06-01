import numpy as np
from sklearn.datasets import load_digits, load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import sys


def fit_perceptron(p: Perceptron, X, Y, times, error=0.1):
    for time in range(1, times + 1):
        p = p.partial_fit(X, Y, classes=np.unique(Y))
        Y_hat = p.predict(X)
        correct = np.equal(Y, Y_hat)
        vals, count = np.unique(correct, return_counts=True)
        index = 0 if not vals[0] else 1
        if len(vals) == 1:
            count = np.append(count, 0)
        false_num = count[index]
        err = false_num/len(correct)
        if err < error:
            return p, time

    return p, times


def make_y(c):
    def y_func(y):
        return 1 if y == c else -1
    return y_func


class OvA:
    def __init__(self):
        self.W = None

    def fit(self, X, Y, times=1000, learning_rate=10**-4):
        self.W = None
        weights = []
        labels_num = len(np.unique(Y))
        for c in range(labels_num):
            y_func = make_y(c)
            w_c, _ = WHALG(X, Y, y_func, times, learning_rate)
            weights.append(w_c)
        self.W = np.column_stack(weights)

    def predict(self, X):
        Y = X@self.W
        return np.argmax(Y, axis=1)


def WHALG(X, Y, y_func, times, learning_rate, error=0.1):
    def sign(x):
        return 1 if x > 0 else -1

    M = X.shape[1]
    w = np.zeros(M)

    for time in range(1, times + 1):
        Y_hat = []
        for x, y in zip(X, Y):
            y = y_func(y)
            y_hat = np.dot(x, w)
            w += learning_rate*(y - y_hat)*x
            Y_hat = np.append(Y_hat, 1 if y_hat > 0 else 0)

        if error is not None:
            correct = np.equal(Y, Y_hat)
            vals, count = np.unique(correct, return_counts=True)
            index = 0 if not vals[0] else 1
            if len(vals) == 1:
                count = np.append(count, 0)
            false_num = count[index]
            err = false_num / len(correct)
            if err < error:
                return w, time

    return w, times


def compare_classification():
    iris = load_iris()
    digits = load_digits()

    lr = 10 ** -4
    epochs = 1000
    perceptron = Perceptron(tol=None, max_iter=epochs, eta0=lr)

    for dataset, name in zip([iris, digits], ["iris", "digits"]):
        X = dataset.data
        Y = dataset.target
        X_ones = add_ones(X)
        lms = OvA()
        lms.fit(X_ones, Y, epochs, lr)
        perceptron.fit(X, Y)
        Y_hat = lms.predict(X_ones)
        accuracy = accuracy_score(Y, Y_hat)
        print(name + " accuracy with LMS: " + str(accuracy))
        Y_hat_p = perceptron.predict(X)
        accuracy_p = accuracy_score(Y, Y_hat_p)
        print(name + " accuracy with Perceptron: " + str(accuracy_p))


def add_ones(X):
    ones = np.ones((X.shape[0], 1))
    return np.append(X, ones, axis=1)


def compare_convergence_times():
    def zero_to_min_one(x): return 1 if x > 0 else -1

    lr = 10 ** -5.9
    epochs = 1000

    #build dataset perceptron is better on
    X, Y = make_blobs(n_samples=400, centers=2,
                           cluster_std=0.2, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=40, cmap='viridis', c=Y)
    ax.grid()
    fig.show()

    X_ones = add_ones(X)

    p1 = Perceptron(eta0=lr)

    p1, times_p1 = fit_perceptron(p1, X, Y, epochs)
    w1, times1 = WHALG(X_ones, Y, zero_to_min_one, sys.maxsize**10-1, lr)

    #build dataset LMS is better on
    X, Y = make_blobs(n_samples=400, centers=2,
                      cluster_std=2, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=40, cmap='viridis', c=Y)
    ax.grid()
    fig.show()

    p2 = Perceptron(eta0=lr)
    p2, times_p2 = fit_perceptron(p2, X, Y, epochs)
    w2, times2 = WHALG(X_ones, Y, zero_to_min_one, sys.maxsize ** 10 - 1, lr)

    return times1, times_p1, times2, times_p2


if __name__ == '__main__':
    compare_classification()
    LMS_time1, P_time1, LMS_time2, P_time2 = compare_convergence_times()




