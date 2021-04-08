# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:39:51 2020

@author: pit
"""
import numpy as np
import pandas as pd
from math import log



file = 'D:\MachineLearningPandaIT\Materials\Heart-Disease-UCI.csv'
def data_load(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False)
    except:
        print('incorrect file path')
        
    return df
df = data_load(file)

def preprocessing(df):
    from sklearn.preprocessing import LabelEncoder
        
    df.columns = map(str.lower, df.columns)
    df.describe(include='all')
    df.isna().sum()
    df = df.apply(LabelEncoder().fit_transform)
    
    
    return df
df = preprocessing(df)


def split():
    from sklearn.model_selection import train_test_split as tts
    
    X = df.drop(columns = 'target').values #dependant variable = attrition
    y = df.loc[:,'target'].values #independent variables
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.35,random_state=0)
    
    return X_train, X_test, y_train, y_test

X_train = split()[0]
X_test = split()[1]
y_train = split()[2]
y_test = split()[3]



def decision_tree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    
    
    clf = DecisionTreeClassifier(splitter='best',
                                 min_samples_split=10,
                                 criterion='entropy',
                                 min_samples_leaf=10,
                                 max_features=13,
                                 max_depth= 5)
    clf.fit(X_train , y_train)
    pred = clf.predict(X_test)    
    
    return clf, pred

clf =   decision_tree(X_train, X_test, y_train, y_test)[0]
pred = decision_tree(X_train, X_test, y_train, y_test)[1]

def display_pred(pred):
    comp = pd.DataFrame({'pred':pred,'test':y_test})
    
    return comp

comp = display_pred(pred)
comp1 = (comp.pred == comp.test)
print('\ncheck if prediction == test values:\n', comp1.value_counts())

#calculate train and test errors
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print('\n','training set accuracy is:', train_acc,'\n',
      'test set accuracy is:', test_acc)


from sklearn import tree
import graphviz    

labels = list(df.columns.values)
labels.remove('target')


data = tree.export_graphviz(clf, out_file=None,
   class_names = ['True','False'],
	filled=True, rounded=True,
    feature_names=labels)
graph = graphviz.Source(data, format="png")
graph_path = "./results/decision_tree_ibm"
graph.render(graph_path) 

