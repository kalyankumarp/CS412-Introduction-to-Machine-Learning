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
    def __init__(self,x=[],y=[],numLayers=2,numNodes=2, numOutputs = 1, eta=0.001,maxIter=10000):
        self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.numOutputs = numOutputs
        self.eta = eta
        self.maxIt = maxIter
        self.weights = [{"weights":np.random.rand(len(x[0]),numNodes)}] #create the weights from the inputs to the first layer
        for i in range(self.nLayers-1):
            self.weights.append({"weights":np.random.rand(numNodes,numNodes)}) #create the random weights between internal layers
            
        self.weights.append({"weights":np.random.rand(numOutputs,numNodes)}) #create weights from final layer to output node
        self.outputs = np.zeros(y.shape)
        self.a = 1 #this is how you define a non-static variable

    def update(self, g =[]):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i]['weights'])):
                for k in range(len(self.weights[i]['weights'][j])):
                    self.weights[i]['weights'][j][k] += self.eta *g[k]
                
    def train(self):
        #Do not change this function header
        
        for e in  range(self.maxIt):
            #print("-----------------------------------------------------------------------------------------------------------------------")
            #print("Epoch " + str(e))
            #print("The weights are " + str(self.weights))
            for i in range(len(self.data)):
                g =[]
                #print(self.data[i])
                #print(self.labels[i])
                g = self.backprop(self.labels[i], self.data[i], self.weights, self.nLayers, self.nNodes) 
                #print(g)

                
                self.update(g)
        #print(self.weights)
        return self.weights


    def predict(self,x=[]):
        #Do not change this function header
        #print('\n')
        #print("Predict Function")
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
        for k in range(self.numOutputs):

            p = np.matmul(prev,np.transpose(self.weights[j+1]["weights"][k]))
            s.append(p)
        q = []
        for i in range(len(s)):
            q.append(sigmoid(s[i]))
        return q

    def feedforward(self, data, weights, nLayers, nNodes):
        #This function is likely to be very helpful, but is not necessary
        #print('\n')
        #print("Feedforward Function")
        r =[]
        prev = data
        r.append(prev)

        t =[]

        for j in range(nLayers):
            l = []
            s = []
            for m in range(nNodes):
                #print(weights[j]["weights"][m])
                s.append(np.matmul(weights[j]["weights"][m], prev))
            
            for k in s:
                l.append(sigmoid(k))
            prev = np.asarray(l)
            #print("Prev = " + str(prev))
            t.append(s)
            r.append(l)
        
        s =[]
        for k in range(self.numOutputs):
            
            #print(weights[j+1]["weights"][k])
            p = np.matmul(prev,np.transpose(weights[j+1]["weights"][k]))
            s.append(p)
        #p = sigmoid(p)
        t.append(s)
        #print("P = "  + str(p))
        #print("t = " + str(t))
        
        
        return t,r

    def backprop(self, d, data, weights, nLayers, nNodes):
        #This function is likely to be very helpful, but is not necessary
        #print('\n')
        #print("BackProp")
        g =[]
        der = []
        t,r = self.feedforward(data, weights, nLayers, nNodes)
        
        q = self.predict(data)
        #r.append(q)
        #print("r = " + str(r))
        #print("Q " + str(q))
        for i in range(nLayers):
            a =[]
            for m in range(nNodes):
                a.append(sigmoid_derivative(t[i][m]))
                #print(a)
            der.append(a)
        diff = []
        s =[]
        for i in range(self.numOutputs):
            s.append(sigmoid_derivative(q[i]))
            diff.append(d - q[i])
        der.append(s)
        #print("der " + str(der))
        #print("diff = " + str(diff))
        for i in reversed(range(len(weights))):
            #print(i)
            layer = weights[i]
            #print("Layer " + str(layer))
            errors = []
            if i != len(weights)-1:
                for j in range(len(layer['weights'])):
                    error = 0
                    #print('\n')
                    #print('\n')
                    #print(j)
                    #print(weights[i + 1]['weights'])
                    for k in range(len(weights[i + 1]['weights'])):
                        #print("k " + str(k))
                        #print(weights[i + 1]['weights'][k][j])
                        #print(weights[i + 1]['s'][k])
                        error = (weights[i + 1]['weights'][k][j] * weights[i + 1]['s'][k])
                    errors.append(error)
                    #print(errors)
            else:
                errors.append(diff[0])
                #print(errors)
            layer['s'] = []
            for j in range(len(layer['weights'])):                
                #print("der[i] " + str(der[i]))
                #print("errors " + str(errors))
                layer['s'].append(errors[j]*der[i][j])
            #print("For layer " + str(i) + " the weights are " + str(self.weights))
            
        #print(self.weights)       
        for j in range(len(self.weights)):
            for k in range(len(self.weights[j]['weights'])):
                for m in range(len(self.weights[j]['weights'][k])):
                    g.append(-(r[j][m])*self.weights[j]['s'][k])
                    
                
            
        
        g = np.asarray(g)   
        #print('\n')
        return g



