#The commented variables are suggestions so change them as appropriate,
#However, do not change the __init__(), train(), or predict(x=[]) function headers
#You may create additional functions as you see fit

import numpy as np
np.random.seed(100)


# Sigmoid is the activation Function
def af(t):
    return 1 / (1 + np.exp(-t))

def af_derivative(p):
    return af(p) * (1 -af(p))

def mse(x, d, w, nLayers, nNodes):
    c = 0
    n = len(x)
    for i in range(0,n):
        prev = x[i] 
            
        for j in range(nLayers):
            u =[]
            p =[]
            for k in range(nNodes):
                t = np.dot(w[j]['weights'][k], prev)
                u.append(t)
            for l in u:
                p.append(af(l))
            p.append(1)
            prev = np.asarray(p)
        y = np.dot(w[nLayers]["weights"][0], prev)
        c += (d[i] - y)**2
            
    return c/n

#np.random.seed(100)
class NeuralNetwork:
    
    def __init__(self,x=[],y=[],numLayers=2,numNodes=2, numOutputs = 1, eta=0.001,maxIter=10000, ep = 0):
        #self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.numOutputs = numOutputs
        self.eta = eta
        self.maxIt = maxIter
        self.ep = ep
        
        #self.g = len(self.data[0])*numLayers + numLayers*numNodes + numNodes*numOutputs
        newdata = []
        for i in range(len(x)):
            newdata.append(np.append(x[i],1))     
        self.data = newdata
        self.weights = [{"weights":np.random.uniform(low =-2, high = 2, size = (self.nNodes, len(self.data[0])))}] 
        for i in range(self.nLayers):
            self.weights.append({"weights":np.random.uniform(low =-2, high = 2, size = (self.nNodes,self.nNodes+1))})             
        if self.nLayers >= 0:            
            self.weights.append({"weights":np.random.uniform(low =-2, high = 2, size = (self.numOutputs,self.nNodes+1))})

        #return self.weights
        #self.train()
        
        
    def train(self):
        
        np.random.shuffle(self.data)
        print(self.data)
        temp_eta = self.eta
#         temp2 = mse(self.data, self.labels, self.weights, self.nLayers+1, self.nNodes)
#         print(temp2)
#         print(f) 
        e = 0
        obj =[]
        epoch = []
        cos = 100000000

        while e < self.maxIt:
            prev = cos
            for i in range(len(self.data)):
                self.backprop(self.labels[i], self.data[i]) 
                for m in range(len(self.weights)):
                    for j in range(len(self.weights[m]['weights'])):
                        for k in range(len(self.weights[m]['weights'][j])):
                            self.weights[m]['weights'][j][k] -= self.eta *self.weights[m]['g'][j][k]


            cos = mse(self.data, self.labels, self.weights, self.nLayers+1,self.nNodes)
            epoch.append(e)
            obj.append(cos)
            e += 1

#             if cos >= prev:
#                 self.eta = 0.9*self.eta
#                 obj =[]
#                 epoch = []
#                 e = 0
#                 obj.append(cos)
#                 epoch.append(e)
#                 cos = 100000000
# #                 self.weights = temp_weights
# #                 print(temp_weights)
                
#                 if self.eta <= 0.00001:                    
#                     self.eta = temp_eta  
# #                     obj =[]
# #                     epoch = []
# #                     e = 0
# #                     cos = 100000000
#                     self.weights = [{"weights":np.random.uniform(low =-1, high = 1, size = (self.nNodes, len(self.data[0])))}] 
#                     for i in range(self.nLayers):
#                         self.weights.append({"weights":np.random.uniform(low =-1, high = 1, size = (self.nNodes,self.nNodes+1))})             
#                     if self.nLayers >= 0:            
#                         self.weights.append({"weights":np.random.uniform(low =-1, high = 1, size = (self.numOutputs,self.nNodes+1))})
                          
#             elif cos < prev:
#                 epoch.append(e)
#                 obj.append(cos)
#                 e += 1
        #print(self.weights)
        return 0.0   
       

    def predict(self,x=[]):
        prev = np.append(x,1)
        for j in range(self.nLayers+1):
            l = []
            s = []
            for m in range(self.nNodes):
                s.append(np.matmul(self.weights[j]["weights"][m], prev))
            for k in s:
                l.append(af(k))
            l.append(1)
            prev = np.asarray(l)
        
        s =[]
        p = np.matmul(self.weights[self.nLayers+1]["weights"][0], prev)
        s.append(p)
        q = af(p)
        return q
        
    def feedforward(self, data):
        r =[]
        prev = data
        r.append(prev)
        t =[]

        for j in range(self.nLayers+1):
            l = []
            s = []
            for m in range(self.nNodes):
                #print(self.weights[j]["weights"][m])
                s.append(np.dot(self.weights[j]["weights"][m], prev))
            
            for k in s:
                l.append(af(k))
            l.append(1)
            prev = np.asarray(l)
            t.append(s)
            r.append(l)
            
        s =[]
        p = np.dot(self.weights[self.nLayers+1]["weights"][0], l)
        s.append(p)
        q = af(p)
        t.append(s)        
        
        return t,r, q

    def backprop(self, d, data):
        der = []
        t,r, q= self.feedforward(data)
#         print('\n')
#         print("t = " + str(t))
#         print('\n')
#         print('r = ' + str(r))  
        #q = self.predict(data)
        
        for i in range(self.nLayers+1):
            a =[]
            for m in range(self.nNodes):
                a.append(af_derivative(t[i][m]))
            der.append(a)
        
        diff = []
        s =[]
        s.append(af_derivative(t[self.nLayers+1][0]))
        diff.append(d - q)
        der.append(s)
        for i in reversed(range(len(self.weights))):
            layer = self.weights[i]
            errors = []
            if i != len(self.weights)-1:
                for j in range(len(layer['weights'])):
                    error = 0
                    
                    for k in range(len(self.weights[i + 1]['weights'])):
                        error += (self.weights[i + 1]['weights'][k][j] * self.weights[i + 1]['s'][k])
                    errors.append(error)
            else:
                errors.append(diff[0])
            layer['s'] = []
            for j in range(len(layer['weights'])):
                layer['s'].append(errors[j]*der[i][j])
#         print('\n')
        #print(weights)
        #print(f)
        for j in range(len(self.weights)):
            layer = self.weights[j]
            layer['g'] = []
            for k in range(len(self.weights[j]['weights'])):
                s =[]
                for m in range(len(self.weights[j]['weights'][k])):
                    s.append(((-(r[j][m])*self.weights[j]['s'][k])*2)/len(self.data))
                layer['g'].append(s)

        return 0.0