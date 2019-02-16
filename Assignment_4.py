# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:44:30 2019

@author: codyj
"""
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
def accuracy(y_test,predicted):
    right = 0
    cm = confusion_matrix(y_test,predicted)    
    for i in range(cm.shape[0]):
        right += cm[i,i]
        
    total = sum(sum(cm))
    print("{}%".format(right/total))
    return(right/total)

data1 = pd.read_csv("data/faithful.csv", index_col = 0)
x = data1.values[:,1].reshape(-1, 1)
y = data1.eruptions

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ft = tree.DecisionTreeRegressor()

ft = ft.fit(x_train,y_train)

predictions = ft.predict(x_test)

mse = mean_squared_error(y_test, predictions)
print("RMSE of predictions with normalization: {}".format(np.sqrt(mse))) 

tree.export_graphviz(ft,out_file='tree.dot')

#One hot encoded variables
data2 = pd.read_csv("data/admissions_one_hot.csv")

x = data2.values[:,1:]
y = data2.values[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ft = tree.DecisionTreeClassifier()
ft = ft.fit(x_train,y_train)

predictions = ft.predict(x_test)
confusion_matrix(y_test,predictions)
accuracy(y_test,predictions)

#Encoded department
data2 = pd.read_csv("data/admissions_encoded.csv"

x = data2.values[:,1:]
y = data2.values[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ft = tree.DecisionTreeClassifier()
ft = ft.fit(x_train,y_train)

predictions = ft.predict(x_test)
confusion_matrix(y_test,predictions)
accuracy(y_test,predictions)

#It does slightly better one hot encoded, but still not great.

data3 = pd.read_csv("data/haireyecolor.csv")

x = data3.values[:,1:]
y = data3.values[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ft = tree.DecisionTreeClassifier()
ft = ft.fit(x_train,y_train)

predictions = ft.predict(x_test)
confusion_matrix(y_test,predictions)
accuracy(y_test,predictions)
