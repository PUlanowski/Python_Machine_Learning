# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:11:59 2020

@author: pit
"""

import numpy as np
import pandas as pd
from distython import HEOM
from sklearn.neighbors import NearestNeighbors


file = 'D:\MachineLearningPandaIT\Materials\imdb.csv'
###############################################################################
def data_preproc(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False) #removing 3% bad records - to refactor
    except:
        print('incorrect file path')
    
    df = df[['tid','title','wordsInTitle', 'type', 'imdbRating',
               'Action','Adult', 'Adventure',
               'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Family', 'Fantasy', 'FilmNoir', 'GameShow', 'History',
               'Horror', 'Music', 'Musical', 'Mystery', 'News', 'RealityTV',
               'Romance', 'SciFi', 'Short', 'Sport', 'TalkShow', 'Thriller',
               'War', 'Western']]
    median = df['imdbRating'].median()
    imdbRating_nan = df['imdbRating'].fillna(median)
    df['imdbRating'] = imdbRating_nan
    del (imdbRating_nan)
    
    df_check_cat = df.loc[:,'Action':'Western']

    check = (0 or 1) in df_check_cat.values
    assert check, 'non-binary values'    
    del (df_check_cat)
    
    df_knn = df.loc[:,'wordsInTitle':'Western']
    df_knn = df_knn.loc[df_knn['type'] == 'video.movie']
    df_words = df_knn.wordsInTitle.astype(str)
    df_knn['wordsInTitle'] = df_words
    df_knn = df_knn.drop(columns = ['type'])
    del (df_words)
    
    return df, median, df_knn   
###############################################################################
df = data_preproc(file)[0]
median = data_preproc(file)[1]
df_knn = data_preproc(file)[2]

###############################################################################
def mov():
    pd.set_option('display.max_rows', None)
    mov = input('\n provide your movie title: ')
    mov = mov.lower()
    
    return mov
###############################################################################
mov = mov()    
###############################################################################
def result(mov):
    result = df[df.title.str.contains(r'(?i)'+ mov, regex= True, na=False)]
    result = result[['title']]
    result = result.reset_index()
    result = result[['title']]
    print(result)
    
    return result
###############################################################################
result = result(mov)
###############################################################################    
def index(result):
    select = input('\n please provide selected index: ')
    select = int(select)
    assert type(select) is int,'please provide integer'
    result = result.iloc[select]
    result = result[0]
    idx = (df.index[df['title'] == result]).values.item()
    print('\n you have just selected:', result,'\n')
 
    expr_list = list()
    
    for i in enumerate(df_knn['wordsInTitle']):
        expr =  mov in i
        #print(expr)
        if expr == True:
            expr = 1
        else:
            expr = 0
        expr_list.append(expr)
        
    df_knn['wordsInTitle'] = expr_list
        
    
    return idx,result
###############################################################################
index_return = index(result)
idx = index_return[0]
result = index_return[1]
del (index_return)
###############################################################################
def k():
    k = input('\n how many ideas you would like to see? :')
    k = int(k)
    return k
###############################################################################
k = k()
idx_list = list()
out_title = list()
###############################################################################
def knn():
    #Heterogeneous Euclidean-Overlap Metric
   
    cat_idx = [0]
    heom_metric = HEOM(df_knn, cat_idx, nan_equivalents = median)
    knn = NearestNeighbors(n_neighbors = k, metric = heom_metric.heom)
    knn.fit(df_knn)
    
    main_point = np.array(df_knn.loc[idx]).reshape(1,-1) 
    knn_result = knn.kneighbors(main_point, n_neighbors = k)
    
    idx_list = list(knn_result[1])
    
    for i in idx_list:
        #print(i)
        out_title.append(df.loc[i,'title'])
    
    
    return out_title
###############################################################################
out_title = knn()
print('\n Your recommended movies similar to',result, 'are:\n\n', out_title)        

