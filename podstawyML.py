# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:36:00 2020

@author: pit
"""

import numpy as np

x = np.arange(20, 40+1)
y = np.arange(10, 50+1, 2)

#robimy przedzial 0-1, dla 0 robimy tak:
x = x - np.min(x)
y = y - np.min(y)

#dla calosci


def normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

assert np.all(normalized(x) == normalized(y))

def mean(x):
    return sum(x) / len(x)

def mean_removed(x):
    return x - mean(x)

def variance(x):
    return np.mean(mean_removed(x)**2)

def std_deviation(x):
    return np.sqrt(variance(x))

def standarise(x):
    return mean_removed(x) / std_deviation(x)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit_transform(x.reshape(-1, 1))

def covariance(x, y):
    return np.mean(mean_removed(x) * mean_removed(y))

def correlation(x, y):
    return covariance(x, y) / np.sqrt(variance(x) * variance(y))

np.cov(np.stack((x, y)), bias = False)
np.cov(np.stack((x, y)), bias = True)




