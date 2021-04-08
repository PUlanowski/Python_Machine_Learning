# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:43:53 2020
                                

@author: pit
"""

import numpy as np
import pandas as pd
from math import log



file = 'D:\MachineLearningPandaIT\Materials\IBM-HR-Employee-Attrition.csv'
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
    
    X = df.drop(columns = 'attrition').values #dependant variable = attrition
    y = df.loc[:,'attrition'].values #independent variables
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.35,random_state=0)
    
    return X_train, X_test, y_train, y_test

X_train = split()[0]
X_test = split()[1]
y_train = split()[2]
y_test = split()[3]



def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10,
                                 min_samples_split=10,
                                 criterion='entropy',
                                 min_samples_leaf=7,
                                 max_depth=5,
                                 random_state=1)
    clf.fit(X_train , y_train)
    pred = clf.predict(X_test)   
    
    return clf, pred

clf =   random_forest(X_train, X_test, y_train, y_test)[0]
pred = random_forest(X_train, X_test, y_train, y_test)[1]
  
def random_forest_sfm(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel as sfm  
    
    clf_sfm = RandomForestClassifier(n_estimators=10,
                                 min_samples_split=10,
                                 criterion='entropy',
                                 min_samples_leaf=7,
                                 max_depth=5,
                                 random_state=1)
    
    sfm = sfm(clf_sfm, threshold=0.11)
    sfm.fit(X_train, y_train)
    X_sfm_train = sfm.transform(X_train)
    X_sfm_test = sfm.transform(X_test)
    
    clf_sfm.fit(X_sfm_train, y_train)
    pred_sfm = clf_sfm.predict(X_sfm_test)
    
    return clf_sfm, pred_sfm, X_sfm_train, X_sfm_test

clf_sfm =   random_forest_sfm(X_train, X_test, y_train, y_test)[0]
pred_sfm = random_forest_sfm(X_train, X_test, y_train, y_test)[1]
X_sfm_train =   random_forest_sfm(X_train, X_test, y_train, y_test)[2]
X_sfm_test = random_forest_sfm(X_train, X_test, y_train, y_test)[3]


def display_pred(pred):
    comp = pd.DataFrame({'pred':pred,'test':y_test})
    
    return comp

comp = display_pred(pred)
comp1 = (comp.pred == comp.test)
print('\ncheck if prediction == test values:\n', comp1.value_counts())

#calculate train and test errors
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

#calculate sfm train and test errors
train_acc_sfm = clf_sfm.score(X_sfm_train, y_train)
test_acc_sfm = clf_sfm.score(X_sfm_test, y_test)

print('\n','training set accuracy is:', train_acc,'\n',
      'test set accuracy is:', test_acc)

print('\n','SFM training set accuracy is:', train_acc_sfm,'\n',
      'SFM test set accuracy is:', test_acc_sfm)


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