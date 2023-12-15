import random
import numpy as np
from torchvision import datasets, transforms
from clearml import Task
import plotly.express as px
from rev import RevBlock
import warnings
from dotenv import load_dotenv
from memory_profiler import profile

load_dotenv()


def load_mnist(path="mnist.npz"):
    with np.load(path) as f:
        x_train, _y_train = f["x_train"], f["y_train"]
        x_test, _y_test = f["x_test"], f["y_test"]

    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    y_train = np.zeros((_y_train.shape[0], 10))
    y_train[np.arange(_y_train.shape[0]), _y_train] = 1

    y_test = np.zeros((_y_test.shape[0], 10))
    y_test[np.arange(_y_test.shape[0]), _y_test] = 1

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()


def sigmoid(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.where(
            x >= 0,  # condition
            1 / (1 + np.exp(-x)),  # For positive values
            np.exp(x) / (1 + np.exp(x)),  # For negative values
        )


def sigmoid_prime(z):
    # Derivative of the sigmoid
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(x):
    s = x - np.max(x)
    exps = np.exp(s)
    return exps / exps.sum(axis=0)


def cross_entropy(x, y):
    N = x.shape[0]
    ce = -np.sum(y * np.log(x)) / N
    return ce


class ReVNet(object):
    def __init__(self, sizes):
        self.sizes = sizes
        # initialize biases and weights with random normal distr.
        # weights are indexed by target node first
        # sizes should be in form (mnist_size, a, a, a, ...,  a, 10)
        # IMPORTANT: use at least one layer of size a -> a
        ### Your code goes here ###
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases1 = [np.random.randn(int(y / 2), 1) for y in sizes[1:]]
        self.biases2 = [np.random.randn(int(y / 2), 1) for y in sizes[1:]]

        self.weights1 = [
            np.random.randn(int(y / 2), int(x / 2))
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        self.weights2 = [
            np.random.randn(int(y / 2), int(x / 2))
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

        ###########################

    def feedforward(self, a):
        # Run the network
        ### Your code goes here ###
        a = a.T
        a1, a2 = np.array_split(a, 2, axis=0)

        revBiasesAndWeights = list(
            zip(self.biases1, self.biases2, self.weights1, self.weights2)
        )[1:-1]

        # first layer (no rev block)
        a1 = sigmoid(np.matmul(self.weights1[0], a1) + self.biases1[0])
        a2 = sigmoid(np.matmul(self.weights2[0], a2) + self.biases2[0])

        for b1, b2, w1, w2 in revBiasesAndWeights:
            a1, a2 = RevBlock(b1, b2, w1, w2).block_forward(a1, a2)

        # last layer (no rev block)
        a1 = softmax(np.matmul(self.weights1[-1], a1) + self.biases1[-1])
        a2 = softmax(np.matmul(self.weights2[-1], a2) + self.biases2[-1])

        return np.concatenate((a1, a2), axis=0)

        ###########################

    def update_mini_batch(self, x_mini_batch, y_mini_batch, eta):
        # Update networks weights and biases by applying a single step
        # of gradient descent using backpropagation to compute the gradient.
        # The gradient is computed for a mini_batch.
        # eta is the learning rate
        ### Your code goes here ###
        nabla_b1, nabla_b2, nabla_w1, nabla_w2 = self.backpropagation(
            x_mini_batch.T, y_mini_batch.T
        )

        self.weights1 = [
            w - (eta / len(x_mini_batch)) * nw for w, nw in zip(self.weights1, nabla_w1)
        ]
        self.weights2 = [
            w - (eta / len(x_mini_batch)) * nw for w, nw in zip(self.weights2, nabla_w2)
        ]

        self.biases1 = [
            b - (eta / len(x_mini_batch)) * nb for b, nb in zip(self.biases1, nabla_b1)
        ]
        self.biases2 = [
            b - (eta / len(x_mini_batch)) * nb for b, nb in zip(self.biases2, nabla_b2)
        ]

        ###########################

    def backpropagation(self, x, y):
        ### Your code goes here ###
        a1, a2 = np.array_split(x, 2, axis=0)
        revBiasesAndWeights = list(
            zip(self.biases1, self.biases2, self.weights1, self.weights2)
        )[1:-1]

        # first layer (no rev block)
        a1 = sigmoid(np.matmul(self.weights1[0], a1) + self.biases1[0])
        a2 = sigmoid(np.matmul(self.weights2[0], a2) + self.biases2[0])

        for b1, b2, w1, w2 in revBiasesAndWeights:
            a1, a2 = RevBlock(b1, b2, w1, w2).block_forward(a1, a2)

        # last layer (no rev block)
        x1, x2 = a1, a2
        a1 = sigmoid(np.matmul(self.weights1[-1], a1) + self.biases1[-1])
        a2 = sigmoid(np.matmul(self.weights2[-1], a2) + self.biases2[-1])

        output = np.concatenate((a1, a2), axis=0)

        # backward pass <- both steps at once
        dLdg = self.cost_derivative(output, y)
        dLdf = np.multiply(dLdg, np.multiply(output, 1 - output))

        dLdf1, dLdf2 = np.array_split(dLdf, 2, axis=0)
        dLdg1 = np.matmul(self.weights1[-1].T, dLdf1)
        dLdg2 = np.matmul(self.weights2[-1].T, dLdf2)

        dLdW1s = [np.matmul(dLdf1, x1.T)]
        dLdW2s = [np.matmul(dLdf2, x2.T)]
        dLdB1s = [np.sum(dLdf1, axis=1).reshape(dLdf1.shape[0], 1)]
        dLdB2s = [np.sum(dLdf2, axis=1).reshape(dLdf2.shape[0], 1)]

        a1, a2 = x1, x2

        for b1, b2, w1, w2 in reversed(
            list(
                zip(
                    self.biases1[1:-1],
                    self.biases2[1:-1],
                    self.weights1[1:-1],
                    self.weights2[1:-1],
                )
            )
        ):
            a1, a2, dLdg1, dLdg2, dLdw1, dLdw2, dLdb1, dLdb2 = RevBlock(
                b1, b2, w1, w2
            ).block_reverse(a1, a2, dLdg1, dLdg2)

            dLdW1s.append(dLdw1)
            dLdW2s.append(dLdw2)
            dLdB1s.append(dLdb1)
            dLdB2s.append(dLdb2)

        # first layer:
        x1, x2 = np.array_split(x, 2, axis=0)
        dLdf1 = np.multiply(dLdg1, np.multiply(a1, 1 - a1))
        dLdf2 = np.multiply(dLdg2, np.multiply(a2, 1 - a2))

        dLdW1s.append(np.matmul(dLdf1, x1.T))
        dLdW2s.append(np.matmul(dLdf2, x2.T))
        dLdB1s.append(np.sum(dLdf1, axis=1).reshape(dLdf1.shape[0], 1))
        dLdB2s.append(np.sum(dLdf2, axis=1).reshape(dLdf2.shape[0], 1))

        return (
            list(reversed(dLdB1s)),
            list(reversed(dLdB2s)),
            list(reversed(dLdW1s)),
            list(reversed(dLdW2s)),
        )
        ###########################

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, x_test_data, y_test_data):
        # Count the number of correct answers for test_data
        pred = np.argmax(self.feedforward(x_test_data), axis=0)
        corr = np.argmax(y_test_data, axis=1).T
        return np.mean(pred == corr)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, task=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                y_mini_batch = y_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                self.update_mini_batch(x_mini_batch, y_mini_batch, eta)
            if test_data:
                accuracy = self.evaluate(x_test, y_test)
                
                if task:
                    task.get_logger().report_scalar(title="accuracy",series="accuracy",value=accuracy,iteration=j)

                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate(x_test, y_test)))
            else:
                print("Epoch: {0}".format(j))


class MomentumNetwork(ReVNet):
    def __init__(self, sizes):
        super().__init__(sizes)
        self.momentum_w1 = [np.zeros(w.shape) for w in self.weights1]
        self.momentum_w2 = [np.zeros(w.shape) for w in self.weights2]
        self.momentum_b1 = [np.zeros(b.shape) for b in self.biases1]
        self.momentum_b2 = [np.zeros(b.shape) for b in self.biases2]


    def update_mini_batch(self, x_mini_batch, y_mini_batch, eta, gamma = 0.9, lmbd = 0.0):
        nabla_b1, nabla_b2, nabla_w1, nabla_w2 = self.backpropagation(
            x_mini_batch.T, y_mini_batch.T)
        
        self.momentum_w1 = [gamma * mw - (eta/len(x_mini_batch[0])) * nw - lmbd * w 
                            for w, nw, mw in zip(self.weights1, nabla_w1, self.momentum_w1)]
        self.momentum_w2 = [gamma * mw - (eta/len(x_mini_batch[0])) * nw - lmbd * w 
                            for w, nw, mw in zip(self.weights2, nabla_w2, self.momentum_w2)]
        self.momentum_b1 = [gamma * mb - (eta/len(x_mini_batch[0])) * nb  - lmbd * b
                            for b, nb, mb in zip(self.biases1, nabla_b1, self.momentum_b1)]
        self.momentum_b2 = [gamma * mb - (eta/len(x_mini_batch[0])) * nb  - lmbd * b
                            for b, nb, mb in zip(self.biases2, nabla_b2, self.momentum_b2)]
        
        self.weights1 = [w + mw
                        for w, mw in zip(self.weights1, self.momentum_w1)]
        self.weights2 = [w + mw
                        for w, mw in zip(self.weights2, self.momentum_w2)]
        self.biases1 = [b + mb
                        for b, mb in zip(self.biases1, self.momentum_b1)]
        self.biases2 = [b + mb
                        for b, mb in zip(self.biases2, self.momentum_b2)]
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, gamma, lmbd, test_data=None, task=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                y_mini_batch = y_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                self.update_mini_batch(x_mini_batch, y_mini_batch, eta)
            if test_data:
                accuracy = self.evaluate(x_test, y_test)
                
                if task:
                    task.get_logger().report_scalar(title="accuracy",series="accuracy",value=accuracy,iteration=j)

                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate(x_test, y_test)))
            else:
                print("Epoch: {0}".format(j)) 


if __name__ == "__main__":
    task = Task.init(project_name="HW1", task_name="ReVNEtMorning")

    hyperparams = {"eta": 1.0, "mini_batch_size": 50, "epochs": 100, "gamma": 0.0, "lmbd": 0.0}

    task.connect(hyperparams)


    network = ReVNet([784, 100, 100, 10])
    network.SGD(
        (x_train, y_train),
        epochs=hyperparams["epochs"],
        mini_batch_size=hyperparams["mini_batch_size"],
        eta=hyperparams["eta"],
        test_data=(x_test, y_test),
        task=task,
    )

    task.close()
