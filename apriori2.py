# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:44:50 2020

@author: pit
"""


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori as ap
from mlxtend.frequent_patterns import association_rules as ar

pd.set_option('display.max_columns', None) # display all columns in df, default = 4


#1/ Wczytaj dane z pliku 

file = 'D:\MachineLearningPandaIT\Materials\e-commerce.csv'

def data_load(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'ISO-8859-1',
                         index_col=False, error_bad_lines = False)#, nrows=1000)
    except:
        print('incorrect file path')
        
    return df
df = data_load(file)

# 2/Wyczyść dane
# 3/ Wykonaj wstępną analizę danych

def data_cleanup(df):
    df.head()
    df.isna().sum()
    df_clean = df.drop(columns = ['InvoiceNo', 'StockCode','InvoiceDate'])
    df_clean.isna().sum()
    df_clean = df_clean.dropna(subset=['Description'])
    df_clean['Value'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean.dtypes
    unique_clients = list(df_clean['CustomerID'].unique())
    uc = len(unique_clients)
    item_counter = df_clean['Description']
    ic = item_counter.value_counts()
    country_counter = df_clean[['Country','Value']]
    cc = country_counter['Country'].value_counts()
    cc_val = country_counter.groupby('Country')['Value'].sum()
    cc_val = cc_val.astype(int)
    
    return df_clean, uc, ic, cc, cc_val
    
df_clean = data_cleanup(df)[0]
uc = data_cleanup(df)[1]
ic = data_cleanup(df)[2]
cc = data_cleanup(df)[3]
cc_val = data_cleanup(df)[4]

ic = ic.nlargest(n = 10)
cc = cc.nlargest(n = 10)
cc_val = cc_val.nlargest(n = 10)

print('\n number of unique clients:', uc)
print('\n item counter top 10: \n', ic)
print('\n country by transactions top 10: \n', cc)
print('\n country by values top 10: \n', cc_val)

### narysuj histogram obrazujący 10 nazw państw z największą liczbą transakcji

def charts():
    import matplotlib.pyplot as plt
    labels1 = cc.index.values
    values1 = cc.values
    labels2 = cc_val.index.values
    values2 = cc_val.values

#1 pie chart with %
    explode = (0.5, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0)
    fig1, ax1 = plt.subplots()
    ax1.pie(values1, explode=explode, labels=labels1, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('% contribution in overall transaction count by Country')
    plt.show()
#1 bar chart 
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels1,values1)
    plt.title('numeric contribution in overall transaction count by Country')
    plt.show()

#2 pie chart with %
    explode = (0.5, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0)
    fig1, ax1 = plt.subplots()
    ax1.pie(values2, explode=explode, labels=labels2, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('% contribution in overall value by Country')
    plt.show()
#2 bar chart 
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels2,values2)
    plt.title('% contribution in overall value by Country')
    plt.show()

#4. Wybierz dane tylko dla jednego kraju (np. Polski)
#5. Przekonwertuj dane do postaci którą przyjmuję algorytm Apriori (DataFrame gdzie indexami są
#numery transakcji a wartościami kolumn True lub False w zależności czy dany produkt wystąpił w
#transakcji).
#6. Wygeneruj listę reguł dla wybranej przez Ciebie wartości min_support. Przejrzyj reguły a następnie
#wybierz 5 Twoim zdaniem najlepszych.
#7. Spróbuj zwiększyć wartość min_support, co się wtedy dzieję z liczbą reguł?
#8. Wypisz wszystkie reguły, których wartość lift jest większa niż 5 i wartość confidence
#jest większa niż 0.8


df_EIRE = df[df['Country'] == 'EIRE']
df_EIRE = df_EIRE[['InvoiceNo', 'Description', 'Quantity']]
df_EIRE = (df_EIRE.groupby(['InvoiceNo', 'Description'])['Quantity'].
               sum().unstack().fillna(0))
df_EIRE[df_EIRE == 0] = False
df_EIRE[df_EIRE != 0] = True

freq_items = ap(df_EIRE,
                min_support = 0.05,
                use_colnames=True)
freq_items.head(10)

rules = ar(freq_items,
           metric = 'lift',
           min_threshold = 1)
rules.head(10)

rules['confidence'].sort_values(ascending = False).head(10)
rules['lift'].sort_values(ascending = False).head(10)
rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.8)]
