#The commented variables are suggestions so change them as appropriate,
#However, do not change the __init__(), train(), or predict(x=[]) function headers
#You may create additional functions as you see fit

import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:

    #Do not change this function header
    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.eta = eta
        self.maxIt = maxIter
        #self.weights = [np.random.rand(len(x[0]),numNodes)] #create the weights from the inputs to the first layer
        #for each of the layers
            #self.weights.append(np.random.rand(numNodes,numNodes) #create the random weights between internal layers
            #self.weights.append(np.random.rand(numNodes,1)) #create weights from final layer to output node
            #self.outputs = np.zeros(y.shape)
            #self.train()
        self.a = 1 #this is how you define a non-static variable

    def train(self):
        #Do not change this function header
        return 0.0

    def predict(self,x=[]):
        #Do not change this function header
        return 0.0

    def feedforward(self):
        #This function is likely to be very helpful, but is not necessary
        return 0.0

    def backprop(self):
        #This function is likely to be very helpful, but is not necessary
        return 0.0


