# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:43:53 2020

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



def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    
    n = 10
    clf = RandomForestClassifier(n_estimators=n,
                                 min_samples_split=10,
                                 criterion='entropy',
                                 min_samples_leaf=10,
                                 max_features=13,
                                 max_depth= 5)
    clf.fit(X_train , y_train)
    pred = clf.predict(X_test)    
    
    return clf, pred, n

clf =   random_forest(X_train, X_test, y_train, y_test)[0]
pred = random_forest(X_train, X_test, y_train, y_test)[1]
n = random_forest(X_train, X_test, y_train, y_test)[2]

def display_pred(pred):
    comp = pd.DataFrame({'pred':pred,'test':y_test})
    
    return comp

comp = display_pred(pred)
comp1 = (comp.pred == comp.test)
print('\ncheck if prediction == test values for',n, 'trees:\n', comp1.value_counts())

#calculate train and test errors
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print('\n','training set accuracy is:', train_acc,'\n',
      'test set accuracy is:', test_acc)

'''
from sklearn import tree
import graphviz    

labels = list(df.columns.values)
labels.remove('attrition')

i_tree = 0
for tree_in_forest in clf.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as data:
        data = tree.export_graphviz(tree_in_forest, out_file=None,
                                    class_names = ['True','False'],
                                    filled=True, rounded=True,
                                    feature_names=labels)
    i_tree = i_tree + 1
    graph = graphviz.Source(data, format="png")
    graph_path = "./results/random_forest_ibm_" + str(i_tree)
    graph.render(graph_path) 

'''