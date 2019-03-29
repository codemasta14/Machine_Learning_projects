# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:19:16 2019

@author: codyj
"""

importndas as pd
from math import exp
from sklearn import datasets
import re
def sigmoid(x):
    return(1/(1+exp(-x))) numpy as np
import random
import pa
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
    def __init__(self,weights):
        self.weights = []
        if data.ndim > 1:
                self.datas= np.concatenate((data,np.ones((len(data),1))*-1),axis=1)
                for i in range(len(self.datas[1])):    
                    self.weights.append(random.uniform(-.1,.1))
        else:
                self.datas= np.append(data,-1)
                for i in range(len(self.datas)):    
                    self.weights.append(random.uniform(-.1,.1))
                          
    def output(self,data):
        return(np.dot(self.datas,self.weights))
        
    def activate(self,data):
            if self.output(data).ndim > 0:
                return(list(map(sigmoid,self.output(data))))
            else:
                return(sigmoid(self.output(data)))
#Creates Neural_net class, which consists of nodes.
            
class neural_net():
    def __init__(self,data,nodes=5,hidden_layers = 1,classification = True):                      
        self.nodes = nodes
        self.hidden_layers = hidden_layers
        self.classification = classification

        self.layer_list = {}
        self.data = {}
        for x in range(self.hidden_layers):
            self.layer_list["layer{}".format(x)]={}
            if x == 0:
                if data.ndim > 1:
                    self.data["input{}".format(x)] = data[0]
                else:
                    self.data["input{}".format(x)] = data
            else:
                self.data["input{}".format(x)] =[]
                for snakes in range(self.nodes):
                    self.data["input{}".format(x)].append(self.layer_list["layer{}".format(x-1)]["node{}".format(snakes)].output(self.data["input{}".format(x-1)]))
            self.data["input{}".format(x)]=np.array(self.data["input{}".format(x)])
            for i in range(self.nodes):
                
                self.layer_list["layer{}".format(x)]["node{}".format(i)] = node(self.data["input{}".format(x)])
        self.data["input{}".format(x+1)]=[]
        self.layer_list["output"] = {}
        
        #Creates final input for the last nodes
        for love in range(self.nodes):
                
            self.data["input{}".format(x+1)].append(self.layer_list["layer{}".format(x)]["node{}".format(love)].output(self.data["input{}".format(x)]))
        self.data["input{}".format(x+1)]=np.array(self.data["input{}".format(x+1)])
        self.output={}
        
        for disaster in range(3):   
            self.layer_list["output"]["node{}".format(disaster)] = node(self.data["input{}".format(x+1)])
            self.output["node{}".format(disaster)] = self.layer_list["output"]["node{}".format(disaster)].activate(self.data["input{}".format(x+1)])
    
    def train(self,data):
        predictions = []
        for i in len(data):
            for layer in self.layer_list
                print(layer)
                    
            
        
        
        
        
    def predict(self,data):
         predictions = []
         for n in data:
             for nodes in self.nodes
             
             
             
 #            predictions.append((int(re.search("\d",max(self.output)).group(0))))
        
        

cody = node(wow[0])
cody.output(wow[0])
cody.activate(wow[0])
keagan = neural_net(wow[0:2],5,3)

keagan.predict(wow[0])     
keagan.output
wow[0:2]
keagan.train(wow[2])
keagan.predict(wow[0:2])
keagan.output

list(map(sigmoid,np.array([1,2,3])))
list(map(max,keagan.output))

