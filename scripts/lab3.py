import random
import numpy as np
from torchvision import datasets, transforms
from PIL import Image


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
        

    def feedforward(self, a):
        # Run the network on a batch
        a = a.T
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a)+b)
        return a

    def update_mini_batch(self, mini_batch, eta):
        # Update networks weights and biases by applying a single step
        # of gradient descent using backpropagation to compute the gradient.
        # The gradient is computed for a mini_batch which is as in tensorflow API.
        # eta is the learning rate
        nabla_b, nabla_w = self.backprop(mini_batch[0].T,mini_batch[1].T)

        self.weights = [w-(eta/len(mini_batch[0]))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch[0]))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # For a single input (x,y) return a pair of lists.
        # First contains gradients over biases, second over weights.
        g = x
        gs = [g] # list to store all the gs, layer by layer
        fs = [] # list to store all the fs, layer by layer
        for b, w in zip(self.biases, self.weights):
            f = np.dot(w, g)+b
            fs.append(f)
            g = sigmoid(f)
            gs.append(g)
        # backward pass <- both steps at once
        dLdg = self.cost_derivative(gs[-1], y)
        dLdfs = []
        for w,g in reversed(list(zip(self.weights,gs[1:]))):
            dLdf = np.multiply(dLdg,np.multiply(g,1-g))
            dLdfs.append(dLdf)
            dLdg = np.matmul(w.T, dLdf)

        dLdWs = [np.matmul(dLdf,g.T) for dLdf,g in zip(reversed(dLdfs),gs[:-1])]
        dLdBs = [np.sum(dLdf,axis=1).reshape(dLdf.shape[0],1) for dLdf in reversed(dLdfs)]
        return (dLdBs,dLdWs)

    def evaluate(self, test_data):
        # Count the number of correct answers for test_data
        pred = np.argmax(self.feedforward(test_data[0]),axis=0)
        corr = np.argmax(test_data[1],axis=1).T
        return np.mean(pred==corr)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                y_mini_batch = y_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                self.update_mini_batch((x_mini_batch, y_mini_batch), eta)
            if test_data:
                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate((x_test, y_test))))
            else:
                print("Epoch: {0}".format(j))



class SoftmaxNetwork(Network):
    def __init__(self, sizes):
        super().__init__(sizes)

    def feedforward(self, a):
        # Run the network on a batch
        a = a.T
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.matmul(w, a)+b)

        # last layer, softmax 
        a = softmax(np.matmul(self.weights[-1], a) + self.biases[-1])

        return a
    
    def cost_derivative(self, output_activations, y):
        return (softmax(output_activations)-y)
    
    def backprop(self, x, y):
        # For a single input (x,y) return a pair of lists.
        # First contains gradients over biases, second over weights.
        g = x
        gs = [g] # list to store all the gs, layer by layer
        fs = [] # list to store all the fs, layer by layer

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            f = np.dot(w, g)+b
            fs.append(f)
            g = sigmoid(f)
            gs.append(g)

        # a trick, last layer without activation, because it is easier to compute
        # the gradient of the cost function with respect to the non activated f straight away

        f = np.dot(self.weights[-1], g) + self.biases[-1]
        fs.append(f)

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
    

class L2Network(SoftmaxNetwork):
    def __init__(self, sizes):
        super().__init__(sizes)

    def update_mini_batch(self, mini_batch, eta, lmbd = 0.0):
        # Here we add L2 regularization, the bigger the lmbd, the bigger the penalty

        # Sum of two derrivaties is a derrivative of a sum and we compute
        # the derrivative of L2 regularization term separately only with respect to weights;
        # d (lmbd/2 * sum(w^2)) / dw = lmbd * w

        nabla_b, nabla_w = self.backprop(mini_batch[0].T,mini_batch[1].T)

        self.weights = [(1-eta*lmbd/len(mini_batch[0]))*w-(eta/len(mini_batch[0]))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch[0]))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        

class MomentumNetwork(SoftmaxNetwork):
    def __init__(self, sizes):
        super().__init__(sizes)
        self.momentum_w = [np.zeros(w.shape) for w in self.weights]
        self.momentum_b = [np.zeros(b.shape) for b in self.biases]

    def update_mini_batch(self, mini_batch, eta, gamma = 0.9, lmbd = 0.0):
        # Introducing momentum, the bigger the gamma, the bigger the momentum

        nabla_b, nabla_w = self.backprop(mini_batch[0].T,mini_batch[1].T)

        self.momentum_w = [gamma * mw - (eta/len(mini_batch[0])) * nw - lmbd * w 
                            for w, nw, mw in zip(self.weights, nabla_w, self.momentum_w)]
        
        self.momentum_b = [gamma * mb - (eta/len(mini_batch[0])) * nb  - lmbd * b
                            for b, nb, mb in zip(self.biases, nabla_b, self.momentum_b)]
    

        self.weights = [w + mw
                        for w, mw in zip(self.weights, self.momentum_w)]
        self.biases = [b + mb
                       for b, mb in zip(self.biases, self.momentum_b)]
        

class AdagradNetwork(MomentumNetwork):
    def __init__(self, sizes):
        super().__init__(sizes)
        self.G_t = [np.zeros(w.shape) for w in self.weights]

        def update_mini_batch(self, mini_batch, eta, gamma = 0.0, lmbd = 0.0):
            # Introducing momentum, the bigger the gamma, the bigger the momentum

            nabla_b, nabla_w = self.backprop(mini_batch[0].T,mini_batch[1].T)

            self.G_w = [G_i + nw**2 for G_i, nw in zip(self.G_w, nabla_w)]
            self.G_b = [G_i + nb**2 for G_i, nb in zip(self.G_b, nabla_b)]

            adapted_lr_w = [eta / np.sqrt(G_i + 1e-8) for G_i in self.G_w]
            adapted_lr_b = [eta / np.sqrt(G_i + 1e-8) for G_i in self.G_b]

            self.momentum_w = [gamma * mw - (lr/len(mini_batch[0])) * nw - lmbd * w 
                                for  w, nw, mw, lr in zip(self.weights, nabla_w, self.momentum_w, adapted_lr_w)]
            
            self.momentum_b = [gamma * mb - (lr/len(mini_batch[0])) * nb  - lmbd * b
                                for b, nb, mb, lr in zip(self.biases, nabla_b, self.momentum_b, adapted_lr_b)]
        

            self.weights = [w + mw
                            for w, mw in zip(self.weights, self.momentum_w)]
            self.biases = [b + mb
                        for b, mb in zip(self.biases, self.momentum_b)]
            

class DropoutNetwork(SoftmaxNetwork):
    def __init__(self, sizes):
        super().__init__(sizes)


    def backprop(self, x, y, p = 0.1):
        # For a single input (x,y) return a pair of lists.
        # First contains gradients over biases, second over weights.
        g = x
        mask = np.random.binomial(1, 1 - p, size=g.shape) # one with probability p
        g = np.multiply(g, mask) * (1 / (1- p)) # apply mask and scale
        gs = [g] # list to store all the gs, layer by layer
        fs = [] # list to store all the fs, layer by layer

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            f = np.dot(w, g)+b
            fs.append(f)
            g = sigmoid(f)
            mask = np.random.binomial(1, 1 - p, size=g.shape) # one with probability p
            g = np.multiply(g, mask) * (1 / (1- p)) # apply mask and scale
            gs.append(g)

        # a trick, last layer without activation, because it is easier to compute
        # the gradient of the cost function with respect to the non activated f straight away

        f = np.dot(self.weights[-1], g) + self.biases[-1]
        fs.append(f)

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
    

def rng_rotate(img : np.ndarray, degrees=(-10, 10)):
    pil_image = Image.fromarray(np.uint8(img.reshape((28,28))*255))
    rotation_transform = transforms.RandomRotation(degrees=degrees)
    rotated_pil_image = rotation_transform(pil_image)
    np_image = np.array(rotated_pil_image)

    return np_image.reshape((28*28))/255.

def rng_shift(img : np.ndarray, shift=(-3, 3)):
    pil_image = Image.fromarray(np.uint8(img.reshape((28,28))*255))
    shift_transform = transforms.RandomAffine(degrees=0, translate=shift)
    shifted_pil_image = shift_transform(pil_image)
    np_image = np.array(shifted_pil_image)

    return np_image.reshape((28*28))/255.


class AugumentedNetwork(SoftmaxNetwork):
    def __init__(self, sizes):
        super().__init__(sizes)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        x_train, y_train = training_data
        if test_data:
            x_test, y_test = test_data
        for j in range(epochs):
            for i in range(x_train.shape[0] // mini_batch_size):
                x_mini_batch = x_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                y_mini_batch = y_train[(mini_batch_size*i):(mini_batch_size*(i+1))]
                for k in range(x_mini_batch.shape[0]):
                    if random.random() > 0.5:
                        x_mini_batch[k] = rng_rotate(x_mini_batch[k])

                self.update_mini_batch((x_mini_batch, y_mini_batch), eta)
            if test_data:
                print("Epoch: {0}, Accuracy: {1}".format(j, self.evaluate((x_test, y_test))))
            else:
                print("Epoch: {0}".format(j))
        
        
if __name__ == "__main__":

    network = AugumentedNetwork([784,100,30,10])
    network.SGD((x_train, y_train), epochs=100, mini_batch_size=100, eta=3.0, test_data=(x_test, y_test))

