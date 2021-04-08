# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:05:17 2020

@author: pit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer as si
import statsmodels.regression.linear_model as lm


#dir(preprocessing)

auto =  pd.read_csv('D:\MachineLearningPandaIT\Materials\\auto-mpg.csv',
                        sep='\s+',  decimal='.', header = None)

labels = pd.Series(['mpg','cylinder','displacement','horsepower','weight',
                    'acceleration', 'model_year', 'origin','car_name'])

auto.columns = labels
auto.dtypes

#changing '?' to np.nan
auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')

#changing to array for SimpeImputer
horsepower = auto['horsepower'].to_numpy()

#fitting SI to array using median
si = si(missing_values= np.nan, strategy='median')
si = si.fit(horsepower.reshape(-1, 1))
horsepower = si.transform(horsepower.reshape(-1, 1))

#putting "fitted" horsepower back to DataFrame
auto['horsepower'] = horsepower

X = auto.iloc[:,1:8].values #independent variables = the rest
y = auto.iloc[:,0].values #dependant variable = mpg

#splitting sets
X_train,X_test,y_train, y_test = \
train_test_split(X,y,test_size=0.3, random_state=0)

#fitting multilinear regression on training set / regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction of test set result
y_pred = regressor.predict(X_test)

#bckward elimination
X = np.append(arr = np.ones((398, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

#mean squared error
mse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('mean squared error is: \n',mse)


#printing some statistical values also using statsmodels
print ('intercept: \n', regressor.intercept_)
print ('coefficients: \n', regressor.coef_)


#visualizing training results
plt.scatter(auto.iloc[0:120,0] , y_pred , color = 'red')
plt.scatter(auto.iloc[0:120,0] , y_test, color = 'blue')
plt.title('pred vs test 120 samples')
plt.xlabel('mpg')
plt.ylabel('collections of regressors')
plt.show()
