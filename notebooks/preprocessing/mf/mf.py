# -*- coding: utf-8 -*-
"""mf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZSqz1td6vTKiPDghzVzkpHuU3cZwnYlD
"""

import pandas as pd
import os
import time
ratings_filename = 'train.csv'


df_ratings = pd.read_csv(ratings_filename,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df_movie_features = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

df_movie_features

ratings_filename_test = 'test.csv'


df_ratings_test = pd.read_csv(ratings_filename_test,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df_movie_features_test = df_ratings_test.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

df_movie_features_test

df_ratings_test[~df_ratings_test.movieId.isin(df_ratings.movieId)]

df_ratings_test[~df_ratings_test.userId.isin(df_ratings.userId)]