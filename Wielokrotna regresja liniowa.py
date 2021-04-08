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
import statsmodels.formula.api as sm


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

x = np.array(auto['mpg']).reshape(-1,1)
y = np.array(auto[['cylinder','displacement','horsepower','weight',
          'acceleration', 'model_year']])

x_train,x_test,y_train, y_test = \
train_test_split(x,y,test_size=0.35, random_state=0)

#constructing regressor

regressor = LinearRegression()
regressor.fit(x_train, y_train)

p = regressor.predict(x_test)
#printing some statistical values also using statsmodels
print ('intercept: \n', regressor.intercept_)
print ('coefficient: \n', regressor.coef_)
#Y = np.append(arr = np.ones((398,1)).astype(int), values = y, axis = 1)
'''
Y_opt = y[:,[0,1,2,3,4,5]]
regressor_ols=sm.ols(endog = x, exog = Y_opt).fit()
'''
#visualizing training results
plt.subplot(2 , 1, 1)
plt.scatter(x_train , y_train[:,2] , color = 'blue')
plt.plot(x_train , regressor.predict(x_train)[:,2], color = 'red')
plt.title('training dataset')
plt.xlabel('mpgTrain')
plt.ylabel('horsepowerTrain')
plt.legend(['Regression line'], loc=2)

#visualizing test results
plt.subplot(2 , 1, 2)
plt.scatter(x_test, y_test[:,2], color = 'blue')
plt.plot(x_test, regressor.predict(x_test)[:,2], color = 'red')
plt.title('test dataset')
plt.xlabel('mpgTest')
plt.ylabel('horsepowerTest')
plt.legend(['Regression line'], loc=2)

plt.show()
