import numpy as np


class LinearRegressionGD:
    def __init__(self, regul=0):
        self.regul = regul
        self.W = None

    def fit(self, X, y, lr=0.01, num_iter=1000):
        # Input validation
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must have the same length and cannot be empty")
        
        # Add bias term to X -> [1 X]
        X = np.hstack([np.ones((len(X), 1)), X])

        # Initialize W to zeros
        self.W = np.zeros(X.shape[1])

        # Use gradient descent to minimize cost function
        for i in range(num_iter):
            # Calculate predicted values
            y_pred = np.dot(X, self.W)

            # Calculate cost function
            cost = np.sum((y_pred - y) ** 2) + self.regul * np.sum(self.W ** 2)

            # Calculate gradients
            gradients = 2 * np.dot(X.T, (y_pred - y)) + 2 * self.regul * self.W

            # Update W
            self.W = self.W - lr * gradients

            if (i % 1000 == 0 ): print(cost)

    def predict(self, X):
        # Add bias term to X
        X = np.hstack([np.ones((len(X), 1)), X])

        # Calculate predicted values
        y_pred = np.dot(X, self.W)
        return y_pred
