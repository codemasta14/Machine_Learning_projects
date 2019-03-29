# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:45:08 2019

@author: codyj
"""
import pandas as pd
from math import exp
from sklearn import datasets
import re
import numpy as np
import random
from sklearn.metrics import confusion_matrix
def accuracy(y_test,predicted):
    right = 0
    cm = confusion_matrix(y_test,predicted)    
    for i in range(cm.shape[0]):
        right += cm[i,i]
        
    total = sum(sum(cm))
    print("{}%".format(right/total))
    return(right/total)
    
    
def sigmoid(x):
    return(1/(1+exp(-x))) 


iris = datasets.load_iris()
wow1 = pd.DataFrame(iris.data,columns  = ["a","b","c","d"])


#standardize data
wow1.a = ((wow1.a-np.mean(wow1.a))/np.std(wow1.a))
wow1.b = ((wow1.b-np.mean(wow1.b))/np.std(wow1.b))
wow1.c = ((wow1.c-np.mean(wow1.c))/np.std(wow1.c))
wow1.d = ((wow1.d-np.mean(wow1.d))/np.std(wow1.d))

wow = wow1.values
random.seed = 3
y = np.array(iris.target)


class node():
    def __init__(self,weights):
        self.weights = []
        for i in range(weights+1):    
            self.weights.append(random.uniform(-.1,.1))
        self.weights = np.array(self.weights)

                          
    def output(self,data):
        if data.ndim >1:
            data = np.concatenate((data,np.ones((len(data),1))*-1),axis=1)
        else:
            data = np.append(data,-1)
        return(np.dot(data,self.weights))
        
    def activate(self,data):
            if self.output(data).ndim > 0:
                return(list(map(sigmoid,self.output(data))))
            else:
                return(sigmoid(self.output(data)))
                


class neural_net():
    def __init__(self,weights,hidden_layers = 3,nodes = 3, outputs = 1, classification = True):
        self.classification = classification
        self.layer_list = {}
        
        for layer in range(hidden_layers):
            self.layer_list["layer{}".format(layer)] ={}
            
            for nubs in range(nodes):
                if layer == 0:
                    self.layer_list["layer{}".format(layer)]["node{}".format(nubs)] = node(weights)
                else:
                    self.layer_list["layer{}".format(layer)]["node{}".format(nubs)] = node(nodes)
        self.layer_list["output"]={}
        for outs in range(outputs):
            self.layer_list["output"]["node{}".format(outs)] = node(nodes)
                
<<<<<<< HEAD
    def train(self,data,y,iterations,learn_rate):
        if data.ndim ==1:
            times = 1
            inputs = data
            target = y
=======
    def train(self,data,y):
        if data.ndim ==1:
            times = 1
            inputs = data
>>>>>>> 22576c2c35e02995408c10f6ce5655857f6aa910
            
        else:
            times = data.shape[0]
            inputs = data[n]
<<<<<<< HEAD
            target = y[n]
        
        for n in range(times):
               
               for i, outputs in enumerate(self.layer_list['output']):
                   activations = self.predict(inputs,False)
                   temp = []
                   temp_activation = []
                   for key, value in activations.items():
                       temp = [key,value]
                       temp_activation.append(temp)
                   activation = temp_activation[i][1]
                   dj = activation*(1-activation)*(activation - target)
                   
                   self.layer_list['output'][outputs].weights = self.layer_list['output'][outputs].weights - learn_rate*dj*activation
               
    
    def predict(self,inputs,outputs = True):
=======
        
        for n in range(times):
               
               print(self.predict(inputs))
               
    
    def predict(self,inputs):
>>>>>>> 22576c2c35e02995408c10f6ce5655857f6aa910
        
        classes= []
        if inputs.ndim ==1:
            times = 1
        else:
            times = inputs.shape[0]
        
        for n in range(times):
            if inputs.ndim != 1:
                data = inputs[n]
            else:
                data = inputs
                
            for layer in self.layer_list:
                if layer != "output":
                    self.layer_list[layer]["output"] = []
                    for i, node in enumerate(self.layer_list[layer]):
                        if node != 'output':
<<<<<<< HEAD
                            self.layer_list[layer]["output"].append(self.layer_list[layer][node].activate(data))
=======
                            self.layer_list[layer]["output"].append(self.layer_list[layer][node].output(data))
>>>>>>> 22576c2c35e02995408c10f6ce5655857f6aa910
                    data = np.array(self.layer_list[layer]["output"])
                    
              
                else:
                    self.layer_list[layer]["output"] = {} 
                    for i, node in enumerate(self.layer_list[layer]):
                        if node != "output":
                            self.layer_list[layer]["output"][node] = self.layer_list[layer][node].activate(data)

            if len(self.layer_list["output"]["output"]) ==1:
                if self.layer_list["output"]["output"]["node0"] <.5:
                    classes.append(0)
                else:
                    classes.append(1)
            else:
                classes.append(int(re.search("\d",max(self.layer_list["output"]["output"])).group(0)))
<<<<<<< HEAD
        cam = self.layer_list['output']['output']
        self.layer_list["output"].pop('output')
        if outputs:
            return(classes)
            
        else:
            return(cam)
            
        
        


cody = neural_net(4,1,1,3)
cody.predict(wow[2],False)

for i in range(3):
    print(cody.layer_list['output']['node{}'.format(i)].weights)

cody.train(wow[1],y[1],1,1)

for i in range(3):
    print(cody.layer_list['output']['node{}'.format(i)].weights)
    
accuracy(cody.predict(wow),y)

cody.layer_list['output']
s={}

s['cody'] = 3
s['cody2'] = 3
s.keys()

s.values()
s.items()

temp = []
dictlist = []
for key, value in cody.predict(wow[2],False).items():
    temp = [key,value]
    dictlist.append(temp)
    
dictlist
wow = [] 
for v in s.values():
    wow.append(v)
=======
            
        return(classes)


cody = neural_net(4,5,4,3)
cody.predict(wow)

accuracy(cody.predict(wow),y)

>>>>>>> 22576c2c35e02995408c10f6ce5655857f6aa910
