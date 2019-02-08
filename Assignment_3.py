# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:45:11 2019

@author: Cody Jorgensen
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.preprocessing import StandardScaler
import random
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

iris = datasets.load_iris()

#my own function for accuracy... I like it.
def accuracy(y_test,predicted):
    right = 0
    cm = confusion_matrix(y_test,predicted)    
    for i in range(cm.shape[0]):
        right += cm[i,i]
        
    total = sum(sum(cm))
    print("{}%".format(right/total))
    return(right/total)


class HardCodedClassifier():
    def __init__(self):
        pass
    
    def accuracy(self,results,y_test):
        right = 0
        cm = confusion_matrix(y_test,results)    
        for i in range(cm.shape[0]):
            right += cm[i,i]    
        total = sum(sum(cm))
        print("{}%".format(right/total))
        
        return(right/total)
    
    def predict(self,k,inputs,regress = True):
            nInputs =np.shape(inputs)[0]
            closest =  np.zeros(nInputs)
            if  regress == True:
                
                #Regression
                 for n in range(nInputs):
                    distances = np.sum((self.data-inputs[n,:])**2,axis = 1)
                    
                    indices = np.argsort(distances,axis = 0)
                    closest[n] = np.mean(self.dataClass[indices[:k]])
            else:
                
                #Classification
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
                                closest[n] = np.argmax(counts)
            return(closest)
    
    def fit(self,data,dataClass):
        self.data = data
        self.dataClass = dataClass
        
#Testing sets for iris
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)


        
#Gaussian classifier.
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)
targets_predicted1 = classifier1.predict(X_test)

print(confusion_matrix(y_test,targets_predicted1))
accuracy(y_test,targets_predicted1)
    
#My classifier
classifier2 = HardCodedClassifier()
classifier2.fit(X_train,y_train)
targets_predicted2 = classifier2.predict(3,X_test,False)

print(confusion_matrix(y_test,targets_predicted2))
classifier2.accuracy(targets_predicted2,y_test)

#KNN classifier that isn't mine
from sklearn.neighbors import KNeighborsClassifier

classifier3 = KNeighborsClassifier(n_neighbors=3)
classifier3.fit(X_train,y_train)
predictions = classifier3.predict(X_test)



#Read in and clean data 1 
colnames1 =["price","maint","doors","persons","lug_boot","safety","class"]
data1 =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",names = colnames1)
data1.isna().any()
data1.describe()
data1.head()
data1.price.unique()

leprice = preprocessing.LabelEncoder()
leprice.fit(["low","med","high","vhigh"])
data1.price = leprice.transform(data1.price)

data1.head()
data1.maint.unique()

data1.maint = leprice.transform(data1.maint)

data1.doors.unique()

ledoors = preprocessing.LabelEncoder()
ledoors.fit(["2","3","4","5more"])
data1.doors = ledoors.transform(data1.doors)

data1.head()
data1.persons.unique()

lepers = preprocessing.LabelEncoder()
lepers.fit(["2","4","more"])
data1.persons = lepers.transform(data1.persons)

data1.head()
data1.lug_boot.unique()

leboot = preprocessing.LabelEncoder()
leboot.fit(["small","med","big"])
data1.lug_boot = leboot.transform(data1.lug_boot)

data1.head()
data1.safety.unique()

lesafe = preprocessing.LabelEncoder()
lesafe.fit(["low","med","high"])
data1.safety = lesafe.transform(data1.safety)

data1.head()
data1["class"].unique()
leclass = preprocessing.LabelEncoder()
leclass.fit(["unacc","acc","good","vgood"])
data1["class"] = leclass.transform(data1["class"])
# DONE CLEANING NOW LETS TRAIN AND THEN PREDICT

data1_x = data1.drop(columns = "class")
data1_y = data1["class"]

x_train, x_test, y_train, y_test = train_test_split(data1_x, data1_y, test_size=0.3, random_state=42)
x_test = x_test.values # tursn from DF to numpy array.
y_test = y_test.values
x_train = x_train.values
y_train = y_train.values
classifier4 = HardCodedClassifier()
classifier4.fit(x_train,y_train)

predictions4 = classifier4.predict(3,x_test,False)

confusion_matrix(predictions4,y_test)
accuracy(predictions4,y_test)

#Read in and clean data 2
colnames2 = ["mpg","cyl","dis","hp","wt","acc","year","origin","model"]
data2 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",names = colnames2,sep = "\s+",na_values = "?")
data2.isna().any()
data2=data2.drop(columns="model")
data2 = pd.get_dummies(data2, columns=["cyl"])
data2.dis = (data2.dis - np.mean(data2.dis))/ np.std(data2.dis)
data2.hp = data2.hp.fillna(data2.hp.mean())
data2.hp = (data2.hp - np.mean(data2.hp))/ np.std(data2.hp)
data2.wt = (data2.wt - np.mean(data2.wt)/ np.std(data2.wt))
data2.acc = (data2.acc - np.mean(data2.acc))/ np.std(data2.acc)
data2.year = (data2.year - np.mean(data2.year))/ np.std(data2.year)

#TIME TO SPLIT AND TRAIN
data2_x = data2.drop(columns = "mpg")
data2_y = data2["mpg"]

x_train, x_test, y_train, y_test = train_test_split(data2_x, data2_y, test_size=0.3, random_state=42)
x_test = x_test.values
y_test = y_test.values
x_train = x_train.values
y_train = y_train.values
classifier5 = HardCodedClassifier()
classifier5.fit(x_train,y_train)

predictions5 = classifier5.predict(3,x_test,True)

mse = mean_squared_error(y_test, predictions5)
print("RMSE of predictions with normalization: {}".format(np.sqrt(mse))) 

#Read in and clean data 3

data3 = pd.read_csv("data/student-mat.csv", sep = ";")

data3.dtypes

scaler.fit()
data3.school.unique()
data3.Pstatus.unique()
data3.address.unique()
data3.Medu.unique()

stuff_to_scale = data3.drop(columns = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"])
scaler = StandardScaler().fit(stuff_to_scale)
data3=pd.get_dummies(data3,columns = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"])
data3 = scaler.transform(stuff_to_scale)


data3_x = data3[:,:-1]
data3_y = data3[:,-1]

x_train, x_test, y_train, y_test = train_test_split(data3_x, data3_y, test_size=0.3, random_state=42)

classifier6 = HardCodedClassifier()
classifier6.fit(x_train,y_train)
predictions6 = classifier6.predict(3,x_test,True)

mse = mean_squared_error(y_test, predictions6)
print("RMSE of predictions with normalization: {}".format(np.sqrt(mse))) 