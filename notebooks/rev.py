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
        return sigmoid(np.matmul(self.weightsF, x) + self.biasesF)
    
    def G(self, x):
        return sigmoid(np.matmul(self.weightsG, x) + self.biasesG)

    def block_forward(self, x1, x2):
        z1 = x1 + self.F(x2)
        y2 = x2 + self.G(z1)
        y1 = z1
        return y1, y2

    def block_reverse(self, y1, y2, dLdy1, dLdy2):
        z1 = y1
        x2 = y2 - self.F(z1)
        x1 = z1 - self.G(x2)

        dLdz1 = dLdy1 + dLdy2 * (self.weightsG.T @ sigmoid_prime(self.weightsG @ z1 + self.biasesG))
        dLdx2 = dLdy2 + dLdz1 * (self.weightsF.T @ sigmoid_prime(self.weightsF @ x2 + self.biasesF))
        dLdx1 = dLdz1

        dLdwF = dLdz1 * sigmoid_prime(self.weightsF @ x2 + self.biasesF) @ x2.T
        dLdwG = dLdy2 * sigmoid_prime(self.weightsG @ z1 + self.biasesG) @ z1.T

        dLdFf = dLdz1 * sigmoid_prime(self.weightsF @ x2 + self.biasesF) 
        dLdGf = dLdz1 * sigmoid_prime(self.weightsG @ z1 + self.biasesG)

        dLdbiasF = np.sum(dLdFf, axis=1).reshape(-1, 1)
        dLdbiasG = np.sum(dLdGf, axis=1).reshape(-1, 1)

        return x1, x2, dLdx1, dLdx2, dLdwF, dLdwG, dLdbiasF, dLdbiasG