# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:01:11 2020

@author: pit
"""

# 1. Wygeneruj zbiór dwuwymiarowych danych o tysi¡cu próbek odznaczajacy sie
# losowosci, którego macierz kowariancji jest bliska i wyrysuj go.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn

x = list(np.random.normal(0, 1, 1000))
y = list(reversed(x))
x= np.array(x)
y = np.array(y)

x = x ** 2
y = y * 2

data = ([x,y])

covMatrix = np.cov(data,bias=True)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()


