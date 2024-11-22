import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.slope * x + self.intercept)
        return y_pred
