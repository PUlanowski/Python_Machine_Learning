# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:02:24 2020

@author: pit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


salaries =  pd.read_csv('D:\MachineLearningPandaIT\Materials\salaries.csv',
                        delimiter = ';', header='infer', decimal=',')
#removing NaN data
salaries = salaries.dropna()
#shorting columns names for being lazy sake
salaries.columns = ['salary','xp']

x = np.array(salaries['salary']).reshape(-1,1)
y = np.array(salaries['xp'])

salary_train,salary_test,xp_train, xp_test = \
train_test_split(x,y,test_size=0.35, random_state=0)

#fitting simple linear regression
regressor = LinearRegression()
regressor.fit(salary_train, xp_train)

#predict salary
p = regressor.predict(salary_test)

#visualizing training results
plt.subplot(2 , 1, 1)
plt.scatter(salary_train , xp_train , color = 'blue')
plt.plot(salary_train , regressor.predict(salary_train), color = 'red')
plt.title('training dataset')
plt.xlabel('salary')
plt.ylabel('xp')
plt.legend(['Regression line'], loc=2)

#visualizing test results
plt.subplot(2 , 1, 2)
plt.scatter(salary_test , xp_test , color = 'blue')
plt.plot(salary_test , regressor.predict(salary_test), color = 'red')
plt.title('test dataset')
plt.xlabel('salary')
plt.ylabel('xp')
plt.legend(['Regression line'], loc=2)

plt.show()

#interceptor(b)nd coefficient or slope(m) for y = mx + b 
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
