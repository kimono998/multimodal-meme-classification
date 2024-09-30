# Multimodal Meme Classification

## Course

This project was completed as part of the **Explainable AI** course.

## Task

The goal of this task is to perform multimodal classification of memes, categorizing them as **Offensive** or **Non-offensive** based on both textual and image features.

## Dataset

The dataset used for this task is the **Multimodal Meme Dataset (MultiOFF)** for Identifying Offensive Content in Image and Text:

- **Reference**: Suryawanshi, S., Chakravarthi, B.R., Arcan, M. and Buitelaar, P.

### Example Meme

- **Text Associated**: "OFFICIAL BERNIE SANDERS DRINKING GAME! Every time The Bernster mentions a free government program, chug somebody else's beer!"

## Image Features

The image features are extracted using the following model:

- **Model**: `openai/clip-vit-base-patch32`

Labels used in image analysis include:
- `image_contains_text`
- `image_contains_offensive_words`
- `image_contains_nationality`
- `image_contains_politics`
- `image_contains_country`
- `image_contains_brand`
- `image_contains_positive_words`
- `image_contains_human_being`
- `image_contains_flag`

## Text Features

For text feature extraction, we utilized topic detection based on embedding similarity:

- **Model**: `all-mpnet-base-v2`

## Feature Engineering Process

The feature engineering process involved several approaches and models. The process balanced the need to capture meaningful aspects of the dataset while avoiding overfitting by focusing too much on aspects specific to the current data.

- **Approach**: Handcrafted prompts
- **Tradeoff**: Between capturing meaningful, generalizable features vs. dataset-specific features.

## Evaluation Metrics

We have used the following metrics to evaluate model performance:

1. **Accuracy**
2. **F1-score**

The analysis was conducted using:
- **All features** (image + text)
- **Text features only**
- **Image features only**
