import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        z1 = np.dot(X, W1) + b1
        a1 = np.maximum(0, z1) # ReLU activation function
        z2 = np.dot(a1, W2) + b2
        # probs = 1 / (1 + np.exp(-z2)) # Sigmoid activation function
        exp_z = np.exp(z2)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs

    def loss(self, X, y):
        probs = self.forward(X)
        correct_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(correct_logprobs)
        return 1.0/len(X) * data_loss

    def train(self, X, y, num_epochs, learning_rate=0.1):
        for epoch in range(num_epochs):
            # Forward propagation
            z1 = np.dot(X, self.params['W1']) + self.params['b1']
            a1 = np.maximum(0, z1) # ReLU activation function
            z2 = np.dot(a1, self.params['W2']) + self.params['b2']
            # probs = 1 / (1 + np.exp(-z2)) # Sigmoid activation function
            exp_z = np.exp(z2)
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(len(X)), y] -= 1
            dW2 = np.dot(a1.T, delta3)
            db2 = np.sum(delta3, axis=0)
            delta2 = np.dot(delta3, self.params['W2'].T) * (a1 > 0) # derivative of ReLU
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Update parameters
            self.params['W1'] -= learning_rate * dW1
            self.params['b1'] -= learning_rate * db1
            self.params['W2'] -= learning_rate * dW2
            self.params['b2'] -= learning_rate * db2

            # Print loss for monitoring training progress
            if epoch % 100 == 0:
                loss = self.loss(X, y)
                print("Epoch {}: loss = {}".format(epoch, loss))
