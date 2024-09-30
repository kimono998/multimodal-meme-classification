# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 02:26:52 2023

@author: Dimitrije
"""

import pandas as pd
import numpy as np

#load files with labels
TRAINING_PATH = 'data/Split Dataset/Training_meme_dataset.csv'
TESTING_PATH = 'data/Split Dataset/Testing_meme_dataset.csv'
VALID_PATH = 'data/Split Dataset/Validation_meme_dataset.csv'

def load_csv(FEAT):
    df = pd.read_csv(FEAT)
    df.rename(columns = {'index':'Image Name'}, inplace = True)
    return df
    
    
def feature_concat(PATH, df_ds, file_name):
    df = pd.read_csv(PATH)
    df.rename(columns = {'image_name':'Image Name'}, inplace = True)
    df['label'] = np.where(df['label'] == 'offensive', 1, 0)
    df_done = pd.merge(df_ds, df, on = 'Image Name')
    df_done.drop(columns = ['sentence'], inplace = True)
    df_done.to_csv(file_name, index = False)

