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
                
    def train(self,data,y):
        if data.ndim ==1:
            times = 1
            inputs = data
            
        else:
            times = data.shape[0]
            inputs = data[n]
        
        for n in range(times):
               
               print(self.predict(inputs))
               
    
    def predict(self,inputs):
        
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
                            self.layer_list[layer]["output"].append(self.layer_list[layer][node].output(data))
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
            
        return(classes)


cody = neural_net(4,5,4,3)
cody.predict(wow)

accuracy(cody.predict(wow),y)

