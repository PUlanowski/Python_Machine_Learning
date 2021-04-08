# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:16:26 2020

@author: pit
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori as ap
from mlxtend.frequent_patterns import association_rules as ar

pd.set_option('display.max_columns', None) # display all columns in df, default = 4
file = 'D:\MachineLearningPandaIT\Materials\example_data.csv'

def data_load(file):
    try:
        df = pd.read_csv(file, sep=';', encoding = 'utf-8', index_col=False,
                error_bad_lines = False)
    except:
        print('incorrect file path')
        
    return df
df = data_load(file)
df.head()
df.dtypes
df = df.loc[:,df.columns != 'id']


freq_items = ap(df,
                min_support = 0.11,
                use_colnames=True)
freq_items.head()

rules = ar(freq_items,
           metric = 'lift',
           min_threshold = 1)
rules.head()

rules['confidence'].sort_values(ascending = False).head(100)
rules['lift'].sort_values(ascending = False).head(100)

pd.set_option('display.max_columns', None)

rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.8)]
