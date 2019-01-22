# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:45:11 2019

@author: Cody Jorgensen
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import random
import math
import numpy as np

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)

def accuracy(y_test,predicted):
    cm = confusion_matrix(y_test,predicted)    
    for i in range(cm.shape[0]):
        right += cm[i,i]
        
    total = sum(sum(cm))
    print("{}%".format(right/total))
    return(right/total)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)



classifier = GaussianNB()
classifier.fit(X_train, y_train)

targets_predicted = classifier.predict(X_test)

print(confusion_matrix(y_test,targets_predicted))

class HardCodedClassifier():
    def __init__(self):
        pass
    
    def predict(self,k,inputs):
            nInputs =np.shape(inputs)[0]
            closest =  np.zeros(nInputs)
                
            for n in range(nInputs):
                distances = np.sum((self.data-inputs[n,:])**2,axis = 1)
                
                indices = np.argsort(distances,axis = 0)
                    
                classes = np.unique(self.dataClass[indices[:k]])
                if len(classes)==1:
                    closest[n] = np.unique(classes)
                else:
                    counts = np.zeros(max(classes)+1)
                    for i in range(k):
                        counts[self.dataClass[indices[i]]] +=1
                        closest[n] = np.max(counts)
            return(closest)
        
    
    def fit(self,data,dataClass):
        self.data = data
        self.dataClass = dataClass
        
            
                    
            
        
    
classifier = HardCodedClassifier()
classifier.fit(X_train,y_train)

targets_predicted = classifier.predict(3,X_test)
print(targets_predicted)
cm = confusion_matrix(y_test,targets_predicted)




from sklearn.neighbors import KNeighborsClassifier


classifier2 = KNeighborsClassifier(n_neighbors=3)
classifier2.fit(X_train,y_train)
predictions = classifier2.predict(X_test)

cm2 = confusion_matrix(y_test,predictions)
print(cm2)
print(accuracy(cm2))
