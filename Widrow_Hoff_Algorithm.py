import numpy as np
from sklearn.datasets import load_digits, load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


def make_y(c):
    def y_func(y):
        return 1 if y == c else 0
    return y_func


class OvA:
    def __init__(self):
        self.W = None

    def fit(self, X, Y, labels_num, times=1000, learning_rate=10**-2):
        self.W = None
        weights = []
        for c in range(labels_num):
            y_func = make_y(c)
            w_c = WHALG(X, Y, y_func, times, learning_rate)
            weights.append(w_c)
        self.W = np.column_stack(weights)

    def predict(self, X):
        Y = X@self.W
        res = []
        for row in Y:
            best_index = np.argmax(row)
            res.append(best_index)
        return res


def WHALG(X, Y, y_func, times, learning_rate):
    N = X.shape[0]
    M = X.shape[1]
    w = np.zeros(M)

    # for t in range(times):
    #     index = np.random.randint(0, N)
    #     x, y = X[index], y_func(Y[index])
    #     y_hat = x.transpose()@w
    #     w += learning_rate*(y - y_hat)*x

    for x, y in zip(X, Y):
        y = y_func(y)
        y_hat = np.dot(x, w)
        w += learning_rate*(y - y_hat)*x

    Y_c = [y_func(y) for y in Y]
    w2,_,_,_ = np.linalg.lstsq(X, Y_c)

    return w


if __name__ == '__main__':
    dataset = load_iris()
    X = dataset.data
    Y = dataset.target
    N = len(dataset.target_names)
    classifier = OvA()
    perceptron = Perceptron()
    classifier.fit(X, Y, N)
    perceptron.fit(X, Y)
    X_cut = X[:2000]
    Y_hat = classifier.predict(X)
    accuracy = accuracy_score(Y[:2000], Y_hat)
    score = perceptron.score(X, Y)


#digits learning rate 10**-5