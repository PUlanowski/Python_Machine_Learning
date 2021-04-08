# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:52:30 2020

@author: pit
"""
import numpy as np
import pandas as pd
from math import log
from sklearn.tree import export_graphviz


list  = [0,0,0,0,1,1,2,3]
#list = [0,1,2,3,4,5,6,7]
#list = range(8)
#list = ['b','a','c','a']

entropy_list = []

def entropy(list):

    label, counts = np.unique(list , return_counts = True)
    p_counts = counts/counts.sum()    
        
    for i in p_counts:
        entropy = i*(np.log2(1/i))
        entropy_list.append(entropy)
        
    entropy = sum(entropy_list) 
    return entropy

entropy =entropy(list)    
print('entropy:' ,entropy)

'''
assert entropy([0, 0, 1, 2]) == 1.5
assert entropy(range(8)) == 3.0
assert entropy(i for i in range(8)) == 3.0
assert entropy('baca') == 1.5
'''