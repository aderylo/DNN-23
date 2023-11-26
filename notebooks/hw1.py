import random
import numpy as np
from torchvision import datasets, transforms
from clearml import Task
import plotly.express as px


def load_mnist(path='mnist.npz'):
    with np.load(path) as f:
        x_train, _y_train = f['x_train'], f['y_train']
        x_test, _y_test = f['x_test'], f['y_test']

    x_train = x_train.reshape(-1, 28 * 28) / 255.
    x_test = x_test.reshape(-1, 28 * 28) / 255.

    y_train = np.zeros((_y_train.shape[0], 10))
    y_train[np.arange(_y_train.shape[0]), _y_train] = 1

    y_test = np.zeros((_y_test.shape[0], 10))
    y_test[np.arange(_y_test.shape[0]), _y_test] = 1

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid
    return sigmoid(z)*(1-sigmoid(z))

def softmax(x):
    s = x - np.max(x)
    exps = np.exp(s)
    return exps / exps.sum(axis = 0)


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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        ###########################

  
    def activated_residual(self, x, weight, bias):
        return sigmoid(np.matmul(weight, x) + bias)

    def block_forward(self, x1, x2, weightF, weightG, biasF, biasG):
        z1 = x1 + self.activated_residual(x2, weightF, biasF)
        y2 = x2 + self.activated_residual(z1, weightG, biasG)
        y1 = z1
        return y1, y2
    
    def feedforward(self, a : np.ndarray):
        # Run the network
        ### Your code goes here ###
        a = a.T
        a1, a2 = np.array_split(a, 2, axis=0)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            wF, wG = np.array_split(w, 2, axis=1)
            bF, bG = np.array_split(b, 2, axis=0)

            a1, a2 = self.block_forward(a1, a2, wF, wG, bF, bG)

        # last layer, softmax or sigmoid? 
        wF, wG = np.array_split(self.weights[-1], 2, axis=1)
        bF, bG = np.array_split(self.biases[-1], 2, axis=0)

        a1 = a1 + sigmoid(np.matmul(wF, a2) + bF)
        a2 = a2 + sigmoid(np.matmul(wG, a1) + bG)

        return np.concatenate((a1, a2), axis=0)


    def update_mini_batch(self, x_mini_batch, y_mini_batch, eta):
        # Update networks weights and biases by applying a single step
        # of gradient descent using backpropagation to compute the gradient.
        # The gradient is computed for a mini_batch.
        # eta is the learning rate
        ### Your code goes here ###
        nabla_b, nabla_w = self.backpropagation(x_mini_batch.T, y_mini_batch.T)

        self.weights = [w-(eta/len(x_mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(x_mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        ###########################

    def block_reverse(self, y1, y2, dLdy1, dLdy2, weightF, weightG, biasF, biasG):
        def dFdx2(x2):
            #f(x) = sigmoid(Wx + b)
            return np.multiply(weightF.T, sigmoid_prime(np.matmul(weightF, x2) + biasF))

        def dGdz1(z1):
            return np.multiply(weightG.T, sigmoid_prime(np.matmul(weightG, z1) + biasG))

        def dFdw(x2):
            return np.matmul(sigmoid_prime(np.matmul(weightF, x2) + biasF), x2.T)

        def dGdw(z1):
            return np.matmul(sigmoid_prime(np.matmul(weightG, z1) + biasG), z1.T)
        
        
        z1 = y1
        x2 = y2 - self.activated_residual(z1, weightG, biasG)
        x1 = z1 - self.activated_residual(x2, weightF, biasF)

        dLdz1 = dLdy1 + np.multiply(dGdz1(z1).T, dLdy2)
        dLdx2 = dLdy2 + np.multiply(dFdx2(x2).T, dLdy2)
        dLdx1 = dLdz1

        dLdwF = np.multiply(dFdw(z1).T, dLdz1)
        dLdwG = np.multiply(dGdw(x2).T, dLdx2)

        return x1, x2, dLdx1, dLdx2, dLdwF, dLdwG
    

    def backpropagation(self, x, y):
        ### Your code goes here ###
        ### f - unactivated, g - activated, L - loss, d - derivative
        ### dLdg - derivative of loss with respect to activated function
        ### dLdf - derivative of loss with respect to unactivated function


        output = self.feedforward(x)
        x1, x2 = np.array_split(output, 2, axis=0)

        dLDout = self.cost_derivative(output, y)
        dLdx1, dLdx2 = np.array_split(dLDout, 2, axis=0)

        dLdWs = [], dLdBs = []

        for w, b in reversed(list(zip(self.weights, self.biases))):
            wF, wG = np.array_split(w, 2, axis=1)
            bF, bG = np.array_split(b, 2, axis=0)

            x1, x2, dLdx1, dLdx2, dLdwF, dLdwG = self.block_reverse(x1, x2, dLdx1, dLdx2, wF, wG, bF, bG)

            dLdWs.append(np.concatenate((dLdwF, dLdwG), axis=1))
            dLdBs.append(np.sum(dLdWs[-1], axis=1).reshape(dLdWs[-1].shape[0],1))

        return (dLdBs,dLdWs)
        ###########################

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self, x_test_data, y_test_data):
        # Count the number of correct answers for test_data
            pred = np.argmax(self.feedforward(x_test_data),axis=0)
            corr = np.argmax(y_test_data,axis=1).T
            return np.mean(pred==corr)


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                y_mini_batch = y_train[i*mini_batch_size:(i*mini_batch_size + mini_batch_size)]
                self.update_mini_batch(x_mini_batch, y_mini_batch, eta)
            if test_data:
                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate(x_test, y_test)))
            else:
                print("Epoch: {0}".format(j))


if __name__ == "__main__":
    network = ReVNet([784,30,10])
    network.SGD((x_train, y_train), epochs=50, mini_batch_size=100, eta=3., test_data=(x_test, y_test))