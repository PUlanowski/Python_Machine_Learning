# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:56:20 2020

@author: pit
"""



import numpy as np
import pandas as pd
from math import log


# 1. czytaj dane z pliku Mall_Customers.csv

file = 'D:\MachineLearningPandaIT\Materials\Mall_Customers.csv'
def data_load(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False)
    except:
        print('incorrect file path')
        
    return df
df = data_load(file)

df.columns = map(str.lower, df.columns)
df.describe(include='all')
df.isna().sum()
df.dtypes

# 2. Wykonaj wstępną analizę danych

import seaborn as sns
import matplotlib.pyplot as plt
'''
plot_income = sns.scatterplot(data = df,
                x = "customerid", y = "annual income (k$)",
                palette = "bright")
plot_income.set_title("Income distr.")

plot_age = sns.scatterplot(data = df,
                x = "customerid", y = "age",
                palette = "bright")
plot_age.set_title("Age distr.")

plot_gender = sns.scatterplot(data = df,
                x = "customerid", y = "gender",
                palette = "bright")
plot_gender.set_title("Gender distr.")

plot_spend = sns.scatterplot(data = df,
                x = "customerid", y = "spending score (1-100)",
                palette = "bright")
plot_spend.set_title("Spending distr.")

plot_spend_gndr = sns.scatterplot(data = df,
                x = "customerid", y = "spending score (1-100)",
                hue = "gender", palette = "bright")
plot_spend_gndr.set_title("Spending/Gender distr.")

plot_income_age = sns.scatterplot(data = df,
                x = "annual income (k$)", y = "age",
                palette = "bright", )
plot_income_age.set_title("Income / Age distr.")
'''

# 3. Wyczyść i przygotuj dane
df.gender = df.gender.replace('Male', 1)
df.gender = df.gender.replace('Female', 0)


# 4. Właściciel sklepu chce wiedzieć jak podzielić jego klientów ze względu na wiek i
# spending score. Utwórz model K-Means i wizualizację która pokaże mu wyniki.
# Użyj metody „Łokcia” (elbow method) do znalezienia odpowiednie K dla klastrowana.
# Nazwij odpowiednio utworzone grupy 
from sklearn.cluster import KMeans

def age_spndg(df):
    # find no of clusters 
    
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters = k, max_iter = 100,
                        n_init = 10, random_state = 0)
        kmeans.fit(df[['age','spending score (1-100)']])
        inertia.append(kmeans.inertia_)
    
    return inertia, K

inertia = age_spndg(df)[0]
K = age_spndg(df)[1]


#check bending point on inertion , result is 
'''
plot_inertia = sns.lineplot(x = K, y = inertia)
plot_inertia.set_title("inertia for age/spndg")
'''


#model
kmeans = KMeans(n_clusters = 5, max_iter = 100, n_init = 10, random_state = 0 )
ymeans = kmeans.fit_predict(df[['age','spending score (1-100)']])

#plots

centers = kmeans.cluster_centers_
'''
sns.scatterplot(x = centers[0], y = centers[1], color = "black" )

plot_age_spndg = sns.scatterplot(data = df,
                x = "age", y = "spending score (1-100)", palette = "bright", )
plot_age_spndg.set_title("Age/Spending distr.")

plot_age_spndg_kmeans = sns.scatterplot(data = df,
                x = "age", y = "spending score (1-100)", palette = "bright", hue = ymeans )
sns.scatterplot(x = centers[:,0], y = centers[:,1], color = "black")
plot_age_spndg_kmeans.set_title("Age/Spending distr.")
'''
# Właściciel sklepu chce również wiedzieć jak można podzielić jego klientów nie
# patrząc na ich wiek a na zarobki. Stwórz model K-Means dla atrybutów zarobków
# i spending score oraz wizualizację która pokaże mu wyniki. Nazwij odpowiednio
# utworzone grupy. Użyj metody „Łokcia” (elbow method) do znalezienia odpowiednie
# K dla klastrowana.

def income_spndg(df):
    # find no of clusters 
    
    inertia2 = []
    K2 = range(1, 11)
    for k in K2:
        kmeans2 = KMeans(n_clusters = k, max_iter = 10,
                        n_init = 10, random_state = 0)
        kmeans2.fit(df[['annual income (k$)','spending score (1-100)']])
        inertia2.append(kmeans2.inertia_)
    
    return inertia2, K2

inertia2 = income_spndg(df)[0]
K2 = income_spndg(df)[1]
'''
plot_inertia = sns.lineplot(x = K2, y = inertia2)
plot_inertia.set_title("inertia for income/spndg")
'''

#model2
kmeans2 = KMeans(n_clusters = 5, max_iter = 100, n_init = 100)
ymeans2 = kmeans2.fit_predict(df[['annual income (k$)','spending score (1-100)']])

#plots2

centers2 = kmeans2.cluster_centers_

plot_income_spndg_kmeans = sns.scatterplot(data = df,
                x = "annual income (k$)", y = "spending score (1-100)",
                palette = "bright", hue = ymeans2 )
sns.scatterplot(x = centers2[:,0], y = centers2[:,1], color = "black")
plot_income_spndg_kmeans.set_title("Income/Spending distr.")



