# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:11:59 2020

@author: pit
"""

import numpy as np
import pandas as pd
from statistics import mode as md

# 1. Załaduj dane treningowe.
'''
Jak czytać dane:
dane to tablica tablic, gdzie każda tablica wewnątrz to dwie informację
[koordynaty sąsiada, klasa sąsiada]. Czytając to możemy powiedzieć tak:
pierwsza osoba ma 37 lat i jest chora, druga osoba ma 12 lat i jest zdrowa
trzecia osoba ma 9 lat i jest zdrowa itd. Wasza funkcja KNN powinna przyjąć
parametr query gdzie query to wiek np. 99. Po przyjęciu tego wieku funkcja ta 
powinna policzyć odległości (tutaj przyda się funkcja z zadania 2 - Euklides) 
99 od wszystkich przykładów testowych (37, 12, 9 …..) ale pamiętając jaką klasę
miał każdy z przykładów. Na końcu po posortowaniu wiecie jakie przykłady są
najbliższe więc wybieracie podane przez użytkownika K i znajdujecie wśród klas
tych przykładów dominantę (zadanie 1) żeby na końcu podać jaką klasę
przypisujecie swojemu query 
'''
disease_data = [[37, 1],
 [12, 0],
 [9, 0],
 [5, 0],
 [64, 1],
 [1, 0],
 [71, 1],
 [6, 0],
 [50, 1],
 [18, 0],
 [11, 0],
 [28, 0],
 [14, 0],
 [68, 1],
 [87, 1],
 [94, 1],
 [86, 1],
 [9, 1],
 [63, 1],
 [22, 1]]

# 2. Zainicjuj K dla wybranej liczby sąsiadów

#k = 10
#sample = 54


#3. Dla każdego przykładu w danych oblicz odległość między przykładem zapytania a
#   bieżącym przykładem z danych a następnie dodaj odległość i indeks przykładu do
#   uporządkowanej kolekcji.
'''3. Napisz algorytm KNN od podstaw'''
def knn(k, sample):
    calc_age = list()
    for i in disease_data:
        age = abs(((sample**2) - ((i[0])**2))**(0.5))
        calc_age.append(age)

    calc_age.sort()
    calc_age = calc_age[:k]
    sick_list = list()

    for i in disease_data:
        sick = (i[1])
        age = abs(((sample**2) - ((i[0])**2))**(0.5))
        if age in calc_age:
            sick_list.append(sick)
# Dla podanego zbioru danych sklasyfikuj czy dana osoba ma problemy zdrowotne
# czy nie.
    dom = md(sick_list)
    if dom == 1:
        print('patient aged: ',sample,'is sick.')
    else:
        print('patient aged: ',sample,'is healthy.')
    
    df = pd.DataFrame(list(zip(calc_age,sick_list)),
                      columns=['dist','sickness'])
    
    return df



''' 1. Napisz funkcję liczącą dominantę (lub znajdź odpowiednik w bibliotece).'''
#licznei manualne przy 2 warosciach
#temp = sum(sick_list) / len(sick_list) > 0.5   
#liczenie za pomoca gotowej funkcji


def dominant():
    dom = md(sick_list)
    return dom

'''2. Napisz funkcję liczącą dystans przy użyciu odległości euklidesowej'''
    
def distance(val1,val2):
    dist = abs(((val2**2) - (val1**2))**(0.5))
    return dist


