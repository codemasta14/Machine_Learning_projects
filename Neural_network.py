# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:19:16 2019

@author: codyj
"""

import numpy as np
import random
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()

    

wow1 = pd.DataFrame(iris.data,columns  = ["a","b","c","d"])


#standardize data
wow1.a = ((wow1.a-np.mean(wow1.a))/np.std(wow1.a))
wow1.b = ((wow1.b-np.mean(wow1.b))/np.std(wow1.b))
wow1.c = ((wow1.c-np.mean(wow1.c))/np.std(wow1.c))
wow1.d = ((wow1.d-np.mean(wow1.d))/np.std(wow1.d))

wow = wow1.values
random.seed = 3

#Creates Node class
class node():
    def __init__(self,data,bias = False):
        self.threshhold=.5
        self.weights = []                  
        for i in range(len(data[1])):    
           self.weights.append(random.uniform(-.1,.1))
        
        self.datas= data
        
        if bias :
            self.datas = np.zeros_like(self.datas)
            self.weights = np.zeros_like(self.weights)-1
            self.threshhold = 0
     
        
        
    def fire(self,threshhold=.5):
        self.threshhold = threshhold
        self.input = np.dot(self.datas,self.weights)
        predictions = []
        for i in range(len(self.input)):
            
            if self.input[i] >=self.threshhold:
                predictions.append(1)
            else:
                predictions.append(0)
        return(predictions)
        
    def adjust_weights(self):
        pass
        
        
        
        
#Creates Neural_net class, which consists of nodes.
class neural_net():
    def __init__(self,data,nodes=5):
        self.nodes = []
        #Creates list of nodes
        for i in range(nodes):
            self.nodes.append(node(data))
        #Creates bias node
        self.nodes.append(node(data,bias = True))
    
    def predict(self,data,threshhold=.5):
        predictions = []
        for k in range(len(data)):
            node_pred = []
            for i in self.nodes:
                node_pred.append(i.fire(threshhold)[k])
            predictions.append(int(np.mean(node_pred)>=.5))
        return(predictions)    
        
    def train(self,data):
        pass

cody = node(wow)
cody.fire()

emma = neural_net(wow)

emma.predict(wow)
