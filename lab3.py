import random
import numpy as np
from torchvision import datasets, transforms


# Let's read the mnist dataset

def load_mnist(path='.'):
    train_set = datasets.MNIST(path, train=True, download=True)
    x_train = train_set.data.numpy()
    _y_train = train_set.targets.numpy()

    test_set = datasets.MNIST(path, train=False, download=True)
    x_test = test_set.data.numpy()
    _y_test = test_set.targets.numpy()

    x_train = x_train.reshape((x_train.shape[0],28*28)) / 255.
    x_test = x_test.reshape((x_test.shape[0],28*28)) / 255.

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


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


class Network(object):
    def __init__(self, sizes):
        # initialize biases and weights with random normal distr.
        # weights are indexed by target node first
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.momentum_w = [np.zeros_like(w) for w in self.weights]
        self.momentum_b = [np.zeros_like(b) for b in self.biases]
        

    def feedforward(self, a):
        # Run the network on a batch
        a = a.T
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.matmul(w, a)+b)

        # last layer with output activation
        a = np.matmul(self.weights[-1], a) + self.biases[-1]
        a = softmax(a)

        return a

    def update_mini_batch(self, mini_batch, eta, gamma, lmbd):
        # Update networks weights and biases by applying a single step
        # of gradient descent using backpropagation to compute the gradient.
        # The gradient is computed for a mini_batch which is as in tensorflow API.
        # eta is the learning rate
        nabla_b, nabla_w = self.backprop(mini_batch[0].T,mini_batch[1].T)

        self.momentum_w = [gamma * mw - (eta/len(mini_batch[0])) * nw - lmbd * w 
                            for w, nw, mw in zip(self.weights, nabla_w, self.momentum_w)]
        
        self.momentum_b = [gamma * mb - (eta/len(mini_batch[0])) * nb  - lmbd * b
                            for b, nb, mb in zip(self.biases, nabla_b, self.momentum_b)]
    

        self.weights = [w + mw
                        for w, mw in zip(self.weights, self.momentum_w)]
        self.biases = [b + mb
                       for b, mb in zip(self.biases, self.momentum_b)]
        

    def backprop(self, x, y):
        # For a single input (x,y) return a pair of lists.
        # First contains gradients over biases, second over weights.
        g = x
        gs = [g] # list to store all the gs, layer by layer
        fs = [] # list to store all the fs, layer by layer

        # forward pass except for last layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            f = np.dot(w, g)+b
            fs.append(f)
            g = sigmoid(f)
            gs.append(g)

        # last layer without activation because it is easier to compute
        # the gradient of the cost function with respect to the non activated f
        f = np.dot(self.weights[-1], g) + self.biases[-1]
        fs.append(f)

        # backward pass <- both steps at once
        dLdf = self.cost_derivative(fs[-1], y)
        dLdfs = [dLdf]
        dLdg = np.matmul(self.weights[-1].T, dLdf)

        for w,g in reversed(list(zip(self.weights[:-1],gs[1:]))):
            dLdf = np.multiply(dLdg,np.multiply(g,1-g))
            dLdfs.append(dLdf)
            dLdg = np.matmul(w.T, dLdf)
        
        dLdWs = [np.matmul(dLdf,g.T) for dLdf,g in zip(reversed(dLdfs),gs)] 
        dLdBs = [np.sum(dLdf,axis=1).reshape(dLdf.shape[0],1) for dLdf in reversed(dLdfs)] 
        return (dLdBs,dLdWs)

    def evaluate(self, test_data):
        # Count the number of correct answers for test_data
        pred = np.argmax(self.feedforward(test_data[0]),axis=0)
        corr = np.argmax(test_data[1],axis=1).T
        return np.mean(pred==corr)

    def cost_derivative(self, output_activations, y):
        return softmax(output_activations) - y

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, gamma=0.0, lmbd=0.0):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                y_mini_batch = y_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                self.update_mini_batch((x_mini_batch, y_mini_batch), eta, gamma, lmbd)
            if test_data:
                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate((x_test, y_test))))
            else:
                print("Epoch: {0}".format(j))


network = Network([784,30,10])
network.SGD((x_train, y_train), epochs=100, mini_batch_size=100, eta=3.0, test_data=(x_test, y_test))

