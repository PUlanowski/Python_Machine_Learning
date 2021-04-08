# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:32:21 2020

@author: pit
"""

import numpy as np

x = np.array([-3, 4, 2, 3])
y = x.astype('uint8') #array([253,   4,   2,   3], dtype=uint8)

x ** 2 #array([ 9, 16,  4,  9], dtype=int32)

x = np.arange(1, 10)
M = x.reshape((3, 3))

pyth_arr = [1.0, 2.0, 3.0]

#series

import numpy as np
import pandas as pd

rng = np.random.RandomState(7)
arr = rng.rand(4)
data = pd.Series(arr)

data = pd.Series(arr, index=['a', 'b', 'c', 'd'])

population_dict = {'Wroclaw' : 641073,
                   'Warszawa':1777972,
                   'Poznan': 536438,
                   'Krakow': 771069}

population = pd.Series(population_dict)

######################################################################

df = pd.DataFrame({'data' : data, 'population': population})
data = pd.Series(arr, index= ['Wroclaw','Warszawa','Poznan','Krakow'])
population = pd.Series(population_dict)

df = pd.DataFrame({'data':data , 'population':population})
df - pd.DataFrame(population, columns=['population'])

data = pd.Series(rng.rand(4))
population - pd.Series(rng.rand(4))
df2 = pd.DataFrame({'data':data , 'population':population})

row = df.iloc[0]

col1 = df.iloc[: ,0]
col2 = df.iloc[: ,1]

df2 = df.iloc[:, 0:2]
df3 = df.loc[df['population'] > 700000]

df.rename(columns={'data' : 'random_data', 'population':'pop_ulation'}, inplace = True) #inplace zamienia w aktualnej tabeli , bez tego trzeba przypisac zmienna i rezultatem jest nowa df
dfchange = df.rename(columns=lambda x: x.lower().replace('_','*'))
dfchange2 = df.rename(columns={'data' : 'random_data', 'population':'pop_ulation'})


