# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 01:30:11 2023

@author: Saswat Dash
"""

from PIL import Image
import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

labels = ['image_contains_text','image_contains_offensive_words','image_contains_nationality',
          'image_contains_politics','image_contains_country','image_contains_brand',
          'image_contains_positive_words','image_contains_human_being','image_contains_flag']

IMG_PATH = 'data/Labelled Images/'
JPG_PATH = 'data/Converted Images/'
model = "openai/clip-vit-base-patch32"

def find_image_list(IMG_PATH,JPG_PATH):
    
    img_list = []
    os.makedirs(JPG_PATH, exist_ok=True)

    for image in os.listdir(IMG_PATH):
        img_list.append(image)
        
    return img_list

def load_model(model):
    
    pipe = pipeline("zero-shot-image-classification", model)
    
    return pipe


def find_image_features(img_list, pipe):
    image_features = []

    for img in tqdm(img_list):
        with Image.open(f'data/Labelled Images/{img}') as image:
                dict_img = pipe(image, candidate_labels = labels)
                p = [val for item in dict_img for key, val in item.items() if key == 'score']
                image_features.append(p)
                
    return image_features

def make_df_img(labels, image_features, img_list):
    
    df_image = pd.DataFrame(image_features, columns = labels)
    df_image[labels] = df_image[labels].applymap(lambda x: f"{x:.3f}")
    df_image['Image Name'] = img_list
    df_image = df_image[['Image Name'] + labels]
    
    return df_image

def save_image_features(df_image):
    df_image.to_csv('image_features_CLIP.csv', index = False)
    
def make_image_features(IMG_PATH,JPG_PATH):
    img_list = find_image_list(IMG_PATH,JPG_PATH)
    pipe = load_model(model)
    image_features = find_image_features(img_list, pipe)
    df_image = make_df_img(labels, image_features, img_list)
    save_image_features(df_image)
    
    


