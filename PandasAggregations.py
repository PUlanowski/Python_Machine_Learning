# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:24:29 2020

@author: pit
"""

import pandas as pd 

print('\n__________________________________________________________________________________________________________\n\n\n')
#1. Wczytaj dane z pliku supermarket_sales.csv. Sprawdź czy wszystkie
#kolumny mają odpowiednie typy. Co z wartościami typu NaN?

salesDF = pd.read_csv('D:\MachineLearningPandaIT\Materials\supermarket_sales.csv',)
'''
print(salesDF.dtypes)
print(salesDF.describe(include='all'))

#to check null's in console only START
print('not null calculation:\n\n' , salesDF.notnull().sum(),
      '\n\nnull calculation:\n\n' , salesDF.isnull().sum())
#to check null's in console only END
'''
salesDF = salesDF.rename(columns=lambda x: x.lower().replace(' ','_'))


#2. Napisz funkcję, która:
#a) Obliczy średnią cenę produktów dla każdej kategorii.

def salesCalculations():
    dfFilterredProdLine = salesDF['product_line'].unique()
        
    for prod in dfFilterredProdLine:
        prodFilterred = salesDF[salesDF['product_line'] == prod]
        prodMean = prodFilterred['total'].mean()
        print(prod,'average price is: ', round(prodMean, 2))

#b) Obliczy sumę pieniędzy wydanych przez klientów.        
    sumAll = salesDF['total'].sum()
    print('\nClients spent sum of:', round(sumAll,2))
#c) Poda która kategoria sprzedawanych produktów jest najbardziej
#popularna.
    mostPopular = salesDF.groupby('product_line').count()
    mostPopular = mostPopular['total']
    mostPopular = mostPopular.sort_values(ascending = False)
    print('\nMost popular category is' ,mostPopular.index[0],
          'with', mostPopular.iloc[0], 'occurences')

#d) Obliczy w którym mieście jest najwięcej klientów kobiet.
    dfFilterredCities = salesDF['city'].unique()
    salesFemaleDF = salesDF[(salesDF.gender == 'Female')]
    allFemaleCitiesVal = pd.Series()    
    allFemaleCitiesCity = pd.Series()    
    
    for city in dfFilterredCities:
        femaleCities = salesFemaleDF[salesFemaleDF['city']==city]
        femaleCities = femaleCities['city'].count()
        allFemaleCitiesVal = allFemaleCitiesVal.append(pd.Series(femaleCities))
        allFemaleCitiesCity = allFemaleCitiesCity.append(pd.Series(city))
    
    allFemaleDF = pd.DataFrame({'qty': allFemaleCitiesVal, 'city':allFemaleCitiesCity})
    #for tidy sake sorting indexes out
    allFeamleDF = allFemaleDF.reset_index(drop=True, inplace=True)
    allFeamleDF = allFemaleDF[allFemaleDF['qty'] == allFemaleDF['qty'].max()]
    
    print('\nmost Females done shopping in:',allFeamleDF['city'].item(), 'with quantity of:',allFeamleDF['qty'].item() )
        
#e) Poda średni ranking dla faktur z przynajmniej 5 sprzedanymi produktami.        
    largeSalesDF = salesDF[salesDF['quantity'] >= 5]
    largeSalesVal = largeSalesDF['rating'].mean()    
        
    print('\naverage rating for at least 5 products on invoice is:', round(largeSalesVal, 4))    
    
#f) Poda dla jakich kategorii średni ranking był najwyższy
    #sorting just for tidy sake in case of further processing
    largeSalesCatDF = largeSalesDF[['product_line','rating']].sort_values(by='product_line')
    largeSalesCatDF = largeSalesDF.groupby(['product_line'])['rating'].mean()
    largeSalesCatDF = largeSalesCatDF.sort_values(ascending = False)
    
    print('\nhighest rating by product category is:\n', round(largeSalesCatDF, 4),'\n\n\n')
    
#function execution:
salesCalculations()    
    
#* Dla każdego miasta znajdź linię produktową, dla której
#średnia liczba sprzedanych produktów jest największa i najmniejsza
#i podaj jej wartość.
    
'''
done outside function some certain variables are duplicated in new function        
'''    
def prodSalesStats():
    prodSalesStats = salesDF[['city','product_line','quantity']]
    prodSalesStats = prodSalesStats.groupby(['city','product_line'])
    prodSalesStats = prodSalesStats.mean()
    prodSalesStats = prodSalesStats.sort_values(by='quantity')
    print('\nlowest quantity is in:',prodSalesStats.index[0], 'with value of:',round(prodSalesStats.iloc[0].item(),4) )
    print('\nhighest quantity is in:',prodSalesStats.index[-1], 'with value of:',round(prodSalesStats.iloc[-1].item(),4),'\n\n\n' )
    print('\n__________________________________________________________________________________________________________')
prodSalesStats()    
    
    
    
    
    
    
    
    