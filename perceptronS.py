import numpy as np
import matplotlib.pyplot as plt


class perceptronSimple:
    def __init__(self, n_inputs, learningRate):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learningRate

    def predict(self, X):
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i] + self.b)
            if(y_est[i] >= 0):
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est

    def fit(self, X, Y, epochs=30):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1, 1))
                self.w += self.eta * (Y[:,i] - y_est) * X[:, i]
                self.b += self.eta * (Y[:,i] - y_est)

    def drawPerceptron2d(self, model):
        w1, w2, b = model.w[0], model.w[1], model.b
        plt.plot([-2, 2], [(1/w2) * (-w1*(-2)-b), (1/w2)*(-w1*(2)-b)])
