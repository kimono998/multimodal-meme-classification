# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 03:08:53 2023

@author: Saswat
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


from image_feature_CLIP import make_image_features
from text_features_topic import make_text_features
from data_prepare import load_csv, feature_concat
from model import prepare_dataframe, prediction, find_metrics,prepare_dataframe_using_text,prepare_dataframe_using_image

IMG_PATH = 'data/Labelled Images/'
JPG_PATH = 'data/Converted Images/'

TRAIN_TEXT_FEAT = 'text_features.csv'
IMAGE_FEAT = 'image_features_CLIP.csv'
TEST_TEXT_FEAT = 'text_features_testing.csv'
VALID_TEXT_FEAT = 'text_features_valid.csv'

TRAINING_PATH = 'data/Split Dataset/Training_meme_dataset.csv'
TESTING_PATH = 'data/Split Dataset/Testing_meme_dataset.csv'
VALID_PATH = 'data/Split Dataset/Validation_meme_dataset.csv'

def main():
    parser = argparse.ArgumentParser(
        description='Meme MultiModal Classification'
    )

    parser.add_argument(
        "--make_features", dest="make_features",
        help="Makes the image and text features",
        action="store_true",
        default=None
    )
    
    parser.add_argument(
        "--train", dest="train",
        help="Trains the XGBoost model",
        action="store_true",
        default=None
    )
    
    args = parser.parse_args()
    
    if args.make_features:
        
        make_image_features(IMG_PATH, JPG_PATH)
        make_text_features(TRAINING_PATH , TRAIN_TEXT_FEAT)
        make_text_features(TESTING_PATH , TEST_TEXT_FEAT)
        make_text_features(VALID_PATH , VALID_TEXT_FEAT)
        
        df_train_features = load_csv(TRAIN_TEXT_FEAT)
        df_image_features = load_csv(IMAGE_FEAT)
        df_test_features =  load_csv(TEST_TEXT_FEAT)
        df_valid_features = load_csv(VALID_TEXT_FEAT)
        
        
        df_training_ds= pd.merge(df_image_features, df_train_features, on = 'Image Name')
        df_testing_ds = pd.merge(df_image_features, df_test_features, on = 'Image Name')
        df_valid_ds= pd.merge(df_image_features, df_valid_features, on = 'Image Name')
        
        feature_concat(TRAINING_PATH, df_training_ds, 'training_dataset.csv')
        feature_concat(TESTING_PATH, df_testing_ds, 'test_dataset.csv')
        feature_concat(VALID_PATH, df_valid_ds, 'validation_dataset.csv')
        
        
    if args.train:
        #change features value here, to get feature specific predictions
        f1 = []
        acc = []
                
        #features=="All":

        X_train, y_train = prepare_dataframe('training_dataset.csv')
        X_test, y_test = prepare_dataframe('test_dataset.csv')
        X_valid, y_valid = prepare_dataframe('validation_dataset.csv')
        eval_set_ = [(X_valid, y_valid)]
        preds, _ = prediction(eval_set_, X_train, y_train, X_test)
        f1_all, acc_all = find_metrics(y_test, preds)
        print(f'f1 score: {f1_all}')
        print(f'Accuracy score: {acc_all}')
        f1.append(f1_all)
        acc.append(acc_all)


        #if features=="Text":

        X_train, y_train = prepare_dataframe_using_text('training_dataset.csv')
        X_test, y_test = prepare_dataframe_using_text('test_dataset.csv')
        X_valid, y_valid = prepare_dataframe_using_text('validation_dataset.csv')
        eval_set_ = [(X_valid, y_valid)]
        preds, _ = prediction(eval_set_, X_train, y_train, X_test)
        f1_text, acc_text = find_metrics(y_test, preds)
        print(f'f1 score: {f1_text}')
        print(f'Accuracy score: {acc_text}')
        f1.append(f1_text)
        acc.append(acc_text)


        #if features=="Image":

        X_train, y_train = prepare_dataframe_using_image('training_dataset.csv')
        X_test, y_test = prepare_dataframe_using_image('test_dataset.csv')
        X_valid, y_valid = prepare_dataframe_using_image('validation_dataset.csv')
        eval_set_ = [(X_valid, y_valid)]
        preds, _ = prediction(eval_set_, X_train, y_train, X_test)
        f1_image, acc_image = find_metrics(y_test, preds)
        print(f'f1 score: {f1_image}')
        print(f'Accuracy score: {acc_image}')
        f1.append(f1_image)
        acc.append(acc_image)

        categories = ['All', 'Text', 'Image']


        # Set up the figure and axis
        fig, ax = plt.subplots()

        # Set the width of each bar
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        bar_positions1 = range(len(categories))
        bar_positions2 = [x + bar_width for x in bar_positions1]

        # Create the bars
        rects1 = ax.bar(bar_positions1, acc, bar_width, label='Accuracy')
        rects2 = ax.bar(bar_positions2, f1, bar_width, label='F1 Score')

        # Add labels, title, and ticks to the graph
        ax.set_xlabel('Categories')
        ax.set_ylabel('Scores')
        ax.set_title('Accuracy and F1 Score Comparison')
        ax.set_xticks([r + bar_width/2 for r in bar_positions1])
        ax.set_xticklabels(categories)

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()


if __name__ == "__main__":
    main()


        
        
        
        
    
    
    

