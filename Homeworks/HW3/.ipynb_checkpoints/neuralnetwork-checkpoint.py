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
        self.weights = [np.random.rand(len(x[0]),numNodes)] #create the weights from the inputs to the first layer
        i = 0
        while i < nLayers -1:
            self.weights.append(np.random.rand(numNodes,numNodes)) #create the random weights between internal layers
            i += 1
        self.weights.append([np.random.rand(numNodes,1)]) #create weights from final layer to output node
        self.outputs = np.zeros(y.shape)
        self.train()
        self.a = 1 #this is how you define a non-static variable

    def train(self):
        #Do not change this function header
        e = 0
        w = self.weights
        t_data = self.data
        iter = self.maxIt
        while e < iter:
            i = 0
            while i < len(t_data):
                g = backprop(self.y, t_data[i], w, self.nLayers, self.nNodes)
                w =  w + self.eta*g
                i += 1
            e += 1
        self.weights = w
        return w


    def predict(self,x=[]):
        #Do not change this function header
        prev = x
        for j in range(self.nLayers):
            l = []
            s = []
            for m in range(self.nNodes):
                s.append(np.matmul(self.weights[j][m], prev))
            for k in s:
                l.append(sigmoid(k))
            prev = np.asarray(l)
        p = np.matmul(prev,self.weights[j+1])
        q = sigmoid(p)
        return q

    def feedforward(self, data, weights, nLayers, nNodes):
        #This function is likely to be very helpful, but is not necessary
        prev = data
        t =[]
        for j in range(nLayers):
            l = []
            s = []
            for m in range(nNodes):
                s.append(np.matmul(weights[j][m], prev))
            for k in s:
                l.append(sigmoid(k))
            prev = np.asarray(l)
            t.append(s)
        p = np.matmul(prev,weights[j+1])
        #q = sigmoid(p)
        
        return t, p

    def backprop(self, d, data, weights, nLayers, nNodes):
        #This function is likely to be very helpful, but is not necessary
        g =[]
        der = []
        for i in range(nLayers):
            p =[]
            for m in range(nNodes):
                p.append(sigmoid_derivative(t[i][m]))
            der.append(p)
        
        t,p = feedforward(data, weights, nLayers, nNodes)
        q = predict(data)
        der.append(sigmoid_derivative(p))
        diff = d - q
        for i in reversed(range(len(weights))):
            layer = weights[i]
            errors = []
            if i != len(weights)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for k in weights[i + 1]:
                        error += (k[j] * k['s'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    errors.append(diff)
            for j in range(len(layer)):
                n = layer[j]
                n['s'] = errors[j] * der[i][j]
        
        for i in weights:
            g.append(i['s'])
            
        g = np.asarray(g)
            
        return g


