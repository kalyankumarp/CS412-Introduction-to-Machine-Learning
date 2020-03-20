#The commented variables are suggestions so change them as appropriate,
#However, do not change the __init__(), train(), or predict(x=[]) function headers
#You may create additional functions as you see fit

import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 -p)

class NeuralNetwork:

    #Do not change this function header
    def __init__(self,x=[],y=[],numLayers=2,numNodes=2, numOutputs = 1, eta=0.001,maxIter=10000):
        self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.numOutputs = numOutputs
        self.eta = eta
        self.maxIt = maxIter
        #self.g = len(self.data[0])*numLayers + numLayers*numNodes + numNodes*numOutputs
        self.weights = [{"weights":np.random.rand(numNodes, len(x[0]))}] #create the weights from the inputs to the first layer
        for i in range(self.nLayers-1):
            self.weights.append({"weights":np.random.rand(numNodes,numNodes)}) #create the random weights between internal layers
            
        if self.nLayers > 0:
            
            self.weights.append({"weights":np.random.rand(numOutputs,numNodes)}) #create weights from final layer to output node
        self.outputs = np.zeros(y.shape)

                
    def train(self):
        #np.random.shuffle(self.data)
        print(self.weights)
        for e in  range(2 *self.maxIt):
            
            for i in range(len(self.data)):
                self.backprop(self.labels[i], self.data[i], self.weights, self.nLayers, self.nNodes) 
                for m in range(len(self.weights)):
                    for j in range(len(self.weights[m]['weights'])):
                        for k in range(len(self.weights[m]['weights'][j])):
                            self.weights[m]['weights'][j][k] -= self.eta *self.weights[m]['g'][j][k]
        return self.weights


    def predict(self,x=[]):
        prev = x
        for j in range(self.nLayers):
            l = []
            s = []
            for m in range(self.nNodes):
                s.append(np.matmul(self.weights[j]["weights"][m], prev))
            for k in s:
                l.append(sigmoid(k))
            prev = np.asarray(l)
        s =[]

        p = np.matmul(self.weights[self.nLayers]["weights"][0], prev)
        s.append(p)
        q = sigmoid(p)
        return q

    def feedforward(self, data, weights, nLayers, nNodes):
        r =[]
        prev = data
        r.append(prev)

        t =[]

        for j in range(nLayers):
            l = []
            s = []
            for m in range(nNodes):
                s.append(np.dot(weights[j]["weights"][m], prev))
            
            for k in s:
                l.append(sigmoid(k))
            prev = np.asarray(l)
            t.append(s)
            r.append(l)
        
        s =[]
        p = np.dot(weights[nLayers]["weights"][0], prev)
        s.append(p)
        #p = sigmoid(p)
        t.append(s)       
        
        return t,r

    def backprop(self, d, data, weights, nLayers, nNodes):
        der = []
        t,r = self.feedforward(data, weights, nLayers, nNodes)
        
        q = self.predict(data)
        for i in range(nLayers):
            a =[]
            for m in range(nNodes):
                a.append(sigmoid_derivative(t[i][m]))
            der.append(a)
        
        diff = []
        s =[]
        s.append(sigmoid_derivative(t[nLayers][0]))
        diff.append(d - q)
        der.append(s)
        for i in reversed(range(len(weights))):
            layer = weights[i]
            errors = []
            if i != len(weights)-1:
                for j in range(len(layer['weights'])):
                    error = 0                    
                    for k in range(len(weights[i + 1]['weights'])):
                        error = (weights[i + 1]['weights'][k][j] * weights[i + 1]['s'][k])
                    errors.append(error)
            else:
                errors.append(diff[0])             
            layer['s'] = []
            for j in range(len(layer['weights'])):
                layer['s'].append(errors[j]*der[i][j])
       
        for j in range(len(self.weights)):
            layer = self.weights[j]
            layer['g'] = []
            for k in range(len(self.weights[j]['weights'])):
                s =[]
                for m in range(len(self.weights[j]['weights'][k])):
                    s.append((-(r[j][m])*self.weights[j]['s'][k])*2)
                layer['g'].append(s)
        #print(self.weights)

        return 0.0



