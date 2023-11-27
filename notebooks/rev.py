import random
import numpy as np
from torchvision import datasets, transforms
from clearml import Task
import plotly.express as px


def sigmoid_prime(z):
    # Derivative of the sigmoid
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class RevBlock(object):
    def __init__(self, biases1, biases2, weights1, weights2):
        self.biasesF, self.biasesG = biases1, biases2
        self.weightsF, self.weightsG = weights1, weights2
    
    def F(self, x):
        # print(self.weightsF.shape, x.shape, self.biasesF.shape)
        return sigmoid(np.matmul(self.weightsF, x) + self.biasesF)
    
    def G(self, x):
        return sigmoid(np.matmul(self.weightsG, x) + self.biasesG)

    def block_forward(self, x1, x2):
        z1 = x1 + self.F(x2)
        y2 = x2 + self.G(z1)
        y1 = z1
        return y1, y2

    def block_reverse(self, y1, y2, dLdy1, dLdy2):
        def dFdx2(x2):
            # Gradient of F with respect to x2
            return np.matmul(self.weightsF.T, sigmoid_prime(self.F(x2)))

        def dGdz1(z1):
            # Gradient of G with respect to z1
            return np.matmul(self.weightsG.T, sigmoid_prime(self.G(z1)))

        def dFdw(x2):
            # Gradient of F with respect to weightsF
            return np.matmul(sigmoid_prime(self.F(x2)), x2.T)

        def dGdw(z1):
            # Gradient of G with respect to weightsG
            return np.matmul(sigmoid_prime(self.G(z1)), z1.T)
        
        z1 = y1
        x2 = y2 - self.F(z1)
        x1 = z1 - self.G(x2)

        dLdz1 = dLdy1 + np.multiply(dLdy2, dGdz1(z1))
        dLdx2 = dLdy2 + np.multiply(dLdz1, dFdx2(x2))
        dLdx1 = dLdz1

    
        dLdwG = sigmoid_prime(self.weightsG @ z1 + self.biasesG) * dLdy2 @ y1.T
        dLdwF = sigmoid_prime(self.weightsF @ x2 + self.biasesF) * dLdz1 @ y2.T

        dLdFf = sigmoid_prime(self.F(x2)) * dLdz1
        dLdGf = sigmoid_prime(self.G(z1)) * dLdx2

        dLdbiasF = np.sum(dLdFf, axis=1).reshape(-1, 1)
        dLdbiasG = np.sum(dLdGf, axis=1).reshape(-1, 1)

        return x1, x2, dLdx1, dLdx2, dLdwF, dLdwG, dLdbiasF, dLdbiasG

    

    def block_reverse2(self, y1, y2, dLdy1, dLdy2):
        def dFdx2(x2):
            return np.matmul(self.weightsF.T, sigmoid_prime(self.F(x2)))

        def dGdz1(z1):
            return np.matmul(self.weightsG.T, sigmoid_prime(self.G(z1)))

        def dFdw(x2):
            return np.matmul(sigmoid_prime(self.F(x2)), x2.T)

        def dGdw(z1):
            return np.matmul(sigmoid_prime(self.G(z1)), z1.T)
        
        z1 = y1
        x2 = y2 - self.F(z1)
        x1 = z1 - self.G(x2)

        # breakpoint()
        # dLdf = np.multiply(dLdg,np.multiply(g,1-g))
        # dLdfs.append(dLdf)
        # dLdg = np.matmul(w.T, dLdf)

        dLdz1 = dLdy1 + np.matmul(dLdy2, dGdz1(z1))
        dLdx2 = dLdy2 + np.matmul(dFdx2(x2).T, dLdz1)
        dLdx1 = dLdz1

        dLdwF = np.multiply(dFdw(z1).T, dLdz1)
        dLdwG = np.multiply(dGdw(x2).T, dLdx2)
        

        dLdFf = np.multiply(dLdy1, sigmoid_prime(self.F(x2)))
        dLdGf = np.multiply(dLdy2, sigmoid_prime(self.G(z1)))

        dLdbiasF = np.sum(dLdFf, axis=1).reshape(dLdFf.shape[0],1)
        dLdbiasG = np.sum(dLdGf, axis=1).reshape(dLdGf.shape[0],1)


        return x1, x2, dLdx1, dLdx2, dLdwF, dLdwG, dLdbiasF, dLdbiasG