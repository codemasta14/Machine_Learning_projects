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


iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)



classifier = GaussianNB()
classifier.fit(X_train, y_train)

targets_predicted = classifier.predict(X_test)
print(targets_predicted)
print(confusion_matrix(y_test,targets_predicted))

class HardCodedClassifier():
    def __init__(self):
        pass
# for my above and beyond, instead of having it return just
#Setosa, I made a regression tree in R, and used the splitting
# parameters from the to classify.        
    def predict(self,x):
        predictions = []
        for i in x:
            if i[2] <1.9:
                predictions.append(0)
            elif i[3] > 1.7:
                predictions.append(2)
            elif i[2] >4.8:
                predictions.append(2)
            else:
                predictions.append(1)
                
        return(predictions)
    
    def fit(self,x,y):
        pass
    
classifier = HardCodedClassifier()
classifier.fit(X_train, y_train)
targets_predicted = classifier.predict(X_test)
print(targets_predicted)
cm = confusion_matrix(y_test,targets_predicted)
print(sum(sum(cm)))

right = 0
def accuracy(y_test,predicted):
    cm = confusion_matrix(y_test,predicted)    
    for i in range(cm.shape[0]):
        right += cm[i,i]
        
    total = sum(sum(cm))
    return(right/total)
