# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:05:17 2020

@author: pit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dir(preprocessing)

df =  pd.read_csv('D:\MachineLearningPandaIT\Materials\\bubble_sort.csv',
                        sep='\s+', delimiter =';')

df = df.sort_values(by=['elements'])

X = df.iloc[:,0:1].values #independent variables = elements
y = df.iloc[:,1:2].values #dependant variable = time

'''#splitting sets - no sence since we got only 10 samples!
X_train,X_test,y_train, y_test = \
train_test_split(X,y,test_size=0.2, random_state = 0)'''


X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


#fitting simple linear regression on training set / regressor
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(fit_intercept = False)
lin_reg.fit(X, y)


#fitting polynomial regression on training set / regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression(fit_intercept = False)
lin_reg_2.fit(X_poly, y)

# visualize linear regression
plt.subplot(2 , 1, 1)
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Lin Reg model check')
plt.xlabel('sample qty')
plt.ylabel('time')
plt.show()
#visualize polynomial regression
plt.subplot(2 , 1, 2)

X_grid = np.arange(min(X), max(X), 10)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),
         color = 'blue')
plt.title('Poly Reg model check')
plt.xlabel('sample qty')
plt.ylabel('time')
plt.show()

#predict sample
sample = np.array(6000)
sample = np.array(sample.reshape(-1,1))

def regression_sample(sample):
    sample = np.array(sample)
    sample = np.array(sample.reshape(-1,1))    
#predict by lenear regression
    linear = lin_reg.predict(sample)

#predict by polynomial regression
    polynomial = lin_reg_2.predict(poly_reg.fit_transform(sample))

#coefficients
#???

    #return linear, polynomial
    print('result for sample of ', sample.item(0), 'is:')
    print('for linear regression: ', "{:.7f}".format(linear.item(0)),
          '\nfor polynomial regression: ', "{:.7f}".format(polynomial.item(0)))
    print('\n')
    print()
    
    ''' try to get unit test for numbers only
    assert sample, 'only numbers accepted'
    AssertionError
    
    
    #task 2
def pred(x : float) -> float:
    return p_regressor.predict(p_features.fit_transform([[x]]))[0]
​
def coeffs(pred):
    c = pred(0)
    a = (pred(2) - 2*pred(1) + c)/2
    b = pred(1) - a - c
    return (a,b,c)
​
a,b,c = coeffs(pred)

'''a