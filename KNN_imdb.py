# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:11:59 2020

@author: pit
"""

import numpy as np
import pandas as pd

file = 'D:\MachineLearningPandaIT\Materials\imdb.csv'
###############################################################################
def data_preproc(file):
    try:
        df = pd.read_csv(file, sep=',', encoding = 'utf-8',  index_col=False,
                error_bad_lines = False) #removing 3% bad records - to refactor
    except:
        print('incorrect file path')
    
    df = df[['tid','title','imdbRating','Action', 'Adult', 'Adventure',
               'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Family', 'Fantasy', 'FilmNoir', 'GameShow', 'History',
               'Horror', 'Music', 'Musical', 'Mystery', 'News', 'RealityTV',
               'Romance', 'SciFi', 'Short', 'Sport', 'TalkShow', 'Thriller',
               'War', 'Western']]
    
    df_check_cat = df[['Action', 'Adult', 'Adventure',
               'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Family', 'Fantasy', 'FilmNoir', 'GameShow', 'History',
               'Horror', 'Music', 'Musical', 'Mystery', 'News', 'RealityTV',
               'Romance', 'SciFi', 'Short', 'Sport', 'TalkShow', 'Thriller',
               'War', 'Western']]
    median = df['imdbRating'].median()
    imdbRating_nan = df['imdbRating'].fillna(median)
    df['imdbRating'] = imdbRating_nan
    del (imdbRating_nan)
    
    check = (0 or 1) in df_check_cat.values
    assert check, 'non-binary values'    
    del (df_check_cat)

    return df   
###############################################################################
df = data_preproc(file)
df.dtypes
df.columns.values
###############################################################################
def result():
    pd.set_option('display.max_rows', None)
    mov = input('\n provide your movie title: ')
    mov = mov.lower()
    result = df[df.title.str.contains(r'(?i)'+ mov, regex= True, na=False)]
    result = result[['title']]
    result = result.reset_index()
    result = result[['title']]
    print(result)
    
    return result
###############################################################################
result = result()
###############################################################################    
def idx(result):
    select = input('\n please provide selected index: ')
    select = int(select)
    assert type(select) is int,'please provide integer'
    result = result.iloc[select]
    result = result[0]
    idx = (df.index[df['title'] == result]).values.item()
    print('\n you have just selected:', result,'\n')
    
    return idx, result
###############################################################################
idx = idx(result)
result = idx[1]
idx = idx[0]
###############################################################################
def k():
    k = input('\n how many ideas you would like to see? :')
    k = int(k)
    return k
###############################################################################
k = k()
dist_list = list()
idx_list = list(df.index.values)
###############################################################################
def knn():
    dist_list = list()
    idx_list = list(df.index.values)
    sample = df.loc[idx,'imdbRating':'Western']
    
    for i in idx_list:
        dist = np.linalg.norm(sample-df.loc[i,'imdbRating':'Western']) 
        dist_list.append(dist)

    out_df = {'idx' : idx_list, 'dist' : dist_list, 'title' : df['title']}
    out_df = pd.DataFrame(out_df)
    out_df = out_df.sort_values(by=['dist'])
    out_df = out_df[0:k]
    out_df = out_df[['title']].values
    
    return out_df
###############################################################################
out_df = knn()
print('\n Your recommended movies similar to',result, 'are:\n', out_df)        

