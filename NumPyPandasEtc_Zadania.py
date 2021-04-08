# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:32:21 2020

@author: pit
"""


#1/Utwórz własny obiekt typu Series i dodaj do niego listę imion z poprzednich zajęć.

import numpy as np
import pandas as pd
from datetime import date

print('\n__________________________________________________________________________________________________________\n\n\n')

with open('D:\MachineLearningPandaIT\Materials\male.txt' , 'r', encoding='iso-8859-2' ) as fileMale:
    fileMale = fileMale.read().split(',')
   
with open('D:\MachineLearningPandaIT\Materials\\female.txt' , 'r', encoding='iso-8859-2') as fileFemale:
    fileFemale = fileFemale.read().split(',')
    
dataMale = pd.Series(fileMale)
dataFemale = pd.Series(fileFemale)


dataNames = dataMale.append(dataFemale)


#2/Utwórz obiekt typu Series, age jako kolejny parametr dla imion (liczby z przedziału 20 – 80)

rng = np.random.RandomState(3)
rng = rng.randint(20,high=80,size=273)

dataAge = (rng.astype(int))

#3/Utwórz obiekt typu Series, sector w jakim pracują wskazani pracownicy
#(IT, Administration,HR, Management, Other)
      
sector = ['IT', 'Administration', 'HR', 'Management', 'Other']
p = [0.2, 0.4, 0.2, 0.05, 0.15]
dataSector = np.random.choice(sector,273, p)

dateToday = date.today()
dateStart = date(2000,1,1)
dateDelta = dateToday - dateStart
dateDelta = dateDelta.days
dateTime =np.datetime64('2000-01-01')
dateTime = dateTime + np.random.randint(1,high=dateDelta,size=273)


#4/Utwórz obiekt typu Series, employment_date (w formacie yyyy/mm/dd) kiedy pracownik
#został zatrudniony.

allData = {'Name' : dataNames,
           'Age' : rng,
           'Sector' : dataSector,
           'EmploymentDate': dateTime}
#5/Stwórz obiekt DataFrame, który będzie zawierał wszystkie poprzednio zdefiniowane obiekty.

dfAll = pd.DataFrame
dfAll = dfAll(allData)

#6/Napisz funkcję, która przyjmie obiekt Series zawierający liczby, a wypiszę do pliku
#podstawowe statystki związane z nim związane (średnia, max i min dla każdej kolumny)

#sample Series START
inputSeries = np.random.RandomState(69).randint(0, high=10000,size=300)
inputSeries = pd.Series(inputSeries)
#sample Series END

def stats(inputSeries):
    if np.dtype(inputSeries) == np.int32:
        #print('TRUE')
        avg = np.average(inputSeries)
        mini = np.min(inputSeries)
        maxi = np.max(inputSeries)
        stats = {'avg' : avg,
                 'min' : mini,
                 'max' : maxi}
  
        print(stats)
        
#7/Napisz funkcję, która policzy średnią wieku w każdym z sektorów
        
    dfAll.dtypes
    dfAll.describe
        
def avgAgeSector(dfAll):
    for job in sector:
        dfFilterredAge = dfAll[dfAll['Sector'] == job]
        mean = dfFilterredAge['Age'].mean()

        print('in sector',job,'average is:',int(mean))
    
    
#8/Napisz funkcję, która policzy jakich sektorów jest najwięcej w grupie wiekowej powyżej n lat.
        
def sectorCountByAge():
    '''this is for user input below

    ageThreshold = input('Type your Age threshold: ')
    if ageThreshold.isdigit():
        print('thanks for correct input')
    else:
        print('please provide number')
    '''    
    ageThreshold = 69 #int(ageThreshold) - for user input, now mocked as 69
    dfFilterredSector = dfAll[dfAll['Age'] >= ageThreshold]
    dfFilterredSector.sort_values(by=['Sector'])
    ageSectorCount = dfFilterredSector.groupby('Sector').count()
    totalSum = dfFilterredSector['Sector'].count()
    ageSectorCount = ageSectorCount.sort_values(by='Name', ascending = False)

    print ('\ntotal result is ', totalSum, ' records (for sanity check only)\n')
    print(ageSectorCount['Name'])    
    
    
#9/Napisz funkcję, która wypisze o jakim imieniu pracownik został zatrudniony najwcześniej.    

def nameLongestEmployment():
    dfFilterredName = dfAll.sort_values(by='EmploymentDate')
    dfFilterredName = dfFilterredName.reset_index(drop=True)
    longestWorkingPerson = dfFilterredName.loc[0]

    print(longestWorkingPerson)

    
###########################
#all functions below    
###########################    
print('********************************************\nstats:')
stats(inputSeries)  
print('********************************************\naverage age in sectors:')
avgAgeSector(dfAll) 
print('********************************************\ncount of people in each sector by given age:')
sectorCountByAge()
print('********************************************\nlongest working employee is:')
nameLongestEmployment()
print('********************************************')

print('\n__________________________________________________________________________________________________________')
###########################
#export DataFrame z pkt. 5    
###########################

dfAll.to_csv('D:\MachineLearningPandaIT\Materials\export.csv', index=False, header=True, encoding='iso-8859-2' )


#10/ Zapisz w pliku w czytelny sposób wszystkie policzone wyniki z poprzednich zadań.


