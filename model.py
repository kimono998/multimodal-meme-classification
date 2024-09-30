# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 02:47:02 2023

@author: Dimitrije and Saswat
"""

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def prepare_dataframe(PATH):
    
    df = pd.read_csv(PATH)
    X = df.drop(columns = ['Image Name','label']).copy()
    y = df.label.copy()
    
    return X,y

def prepare_dataframe_using_image(PATH):
    
    df = pd.read_csv(PATH)
    X = df.drop(columns = ['Image Name','label']).copy()
    
    y = df.label.copy()
    X = df.drop(columns = ['Image Name','derogatory_words', 'nationality_mention', 'political_speech', 
            'country_mention', 'human_mention', 'weapon_mention', 'violence_mention', 
            'education', 'information', 'social_media', 'sexuality', 'race_ethnicity', 
            'economy', 'technology', 'label'])
    
    return X,y



def prepare_dataframe_using_text(PATH):
    
    df = pd.read_csv(PATH)
    X = df.drop(columns = ['Image Name','label']).copy()
    
    y = df.label.copy()
    X = df.drop(columns = ['Image Name','image_contains_text','image_contains_offensive_words','image_contains_nationality',
          'image_contains_politics','image_contains_country','image_contains_brand',
          'image_contains_positive_words','image_contains_human_being','image_contains_flag','label'])
    
    return X,y

def prediction(eval_set_, X_train, y_train, X_test):
    
    bst = XGBClassifier(n_estimators=100, max_depth=7, learning_rate = 0.19, objective='binary:logistic')
    bst.fit(X_train, y_train)
    preds = bst.predict(X_test)
    
    return preds, bst


def find_metrics(y_test, preds):
    
    f1 = f1_score(y_test, preds, average='macro')
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy, f1
    
    
