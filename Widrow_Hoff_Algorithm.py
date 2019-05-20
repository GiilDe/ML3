import numpy as np
from sklearn.datasets import load_digits


def make_y(c):
    def y(x_index, data):
        return 1 if data.target[x_index] == c else -1
    return y


class OvA:
    def __init__(self):
        self.W = None

    def fit(self, data, times=1000, learning_rate=10**-3):
        self.W = None

        for c in data.target_names:
            y = make_y(c)
            w_c = WHALG(data, y, times, learning_rate)
            if self.W is None:
                self.W = np.array(w_c).reshape(-1, 1)
            else:
                self.W = np.append(self.W, w_c, axis=1)

    def predict(self, X):
        Y = X@self.W
        res = []
        for row in Y:
            r = max(row, key=lambda x: x-1 if x >= 0 else x+1)
            res.append(1 if r > 0 else -1)
        return res


def WHALG(dataset, y_func, times, learning_rate):
    N = dataset.data.shape[0]
    M = dataset.data.shape[1]
    w = np.zeros(M)

    for t in range(times):
        ind = np.random.randint(0, N)
        x = dataset.data[ind]
        y_hat = x.transpose()@w
        y = y_func(ind, dataset)
        w += learning_rate*(y - y_hat)*x

    return w


if __name__ == '__main__':
    digits = load_digits()
    classifier = OvA()
    classifier.fit(digits)

