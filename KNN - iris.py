# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:54:07 2020

@author: pit
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix



file = 'D:\MachineLearningPandaIT\Materials\iris.data.csv'
###############################################################################
def data_load(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False)
    except:
        print('incorrect file path')
        
    return df
###############################################################################
df = data_load(file)

df.dtypes
df.head(10)
k = input('\n please provide k: ')
k = int(k) #best k = 7
###############################################################################
def split(df):
    
    X = df.iloc[:,0:4].values #independent variables
    y = df.iloc[:,4].values #dependant variable = mpg
    
    X_train, X_test, y_train, y_test = tss(X,y,test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test

###############################################################################
X_train = split(df)[0]
X_test = split(df)[1]
y_train = split(df)[2]
y_test = split(df)[3]
###############################################################################
def knc_predict(X_train, X_test, y_train):
    knc = KNeighborsClassifier(n_neighbors = k)
    knc.fit(X_train, y_train)
    predict = knc.predict(X_test)

    return predict
###############################################################################
predict = knc_predict(X_train, X_test, y_train)

#confusion matrix
cm = confusion_matrix(y_test, predict)
print('confusion matrix:\n', cm)
#evaluate accuracy
print("accuracy:\n {0}".format(accuracy_score(y_test, predict))) #accuracy
#precision, recall
print('precision:\n', precision_score(y_test, predict, average=None)) #array of factor for each class
print('recall:\n',recall_score(y_test, predict, average=None)) #array of factor for each class

###############################################################################
#plotting nice confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                class_names=['setosa',
                                             'versicolor',
                                             'virginica']) 
plt.show()