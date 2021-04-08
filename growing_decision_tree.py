# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:15:34 2020

@author: pit
"""

import numpy as np
import pandas as pd
from math import log
import seaborn as sns

file = 'D:\MachineLearningPandaIT\Materials\iris.data.csv'
def data_load(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False)
    except:
        print('incorrect file path')
        
    return df
df = data_load(file)

entropy_list = []

def entropy_L(list):

    label, counts = np.unique(list , return_counts = True)
    p_counts = counts/counts.sum()    
        
    for i in p_counts:
        entropy = i*(np.log2(1/i))
        entropy_list.append(entropy)
        
    entropy = sum(entropy_list) 
    return entropy


assert entropy_L([0, 0, 1, 2]) == 1.5


entropy = entropy_L(list)    
print('entropy:' ,entropy)

'''
sns.pairplot(df)
sns.pairplot(df, hue="class")
'''

iris_s = df[df.loc[:,'class'] == 'Iris-setosa']
iris_v = df[df.loc[:,'class'] == 'Iris-virginica']
iris_c = df[df.loc[:,'class'] == 'Iris-versicolor']

isd = iris_s.describe()
ivd = iris_v.describe()
icd = iris_c.describe()

df['petal_surface'] = df.loc[:,'petal_length'] * df.loc[:,'petal_width']
df['sepal_surface'] = df.loc[:,'sepal_length'] * df.loc[:,'sepal_width']
sns.scatterplot(data = df , y = 'sepal_surface', x = 'petal_surface',alpha=0.7, color = 'g', hue = 'class')


