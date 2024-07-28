# Sign Language to Sinhala Text Converter

This project aims to convert sign language symbols into Sinhala text. It uses a sequence of frames (video) for each sign language symbol. 

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Collection](#data-collection)
- [Model and Training](#model-and-training)
- [Results](#results)


## Overview

This project was developed to assist in converting sign language symbols into Sinhala text, providing a valuable tool for communication and translation. The system uses video sequences to identify and classify sign language symbols.

## Dataset

The dataset contains 26 sign symbols, each with 15 samples. The symbols are captured in video format, where a sequence of frames represents each symbol.

'ආසන්න වෙලා', 'ඉක්මනින්', 'ඉබ්බා', 'ඉලක්කයට', 'ඒ නිසා', 'ඔබ', 'කපුටාට', 'කපුටෙක්', 'කෑමක්', 'කියලා', 'කේජු කෑල්ලක්', 'දවසක්', 'දිනුවා', 'පරක්කු උණා', 'බඩගින්නෙන්', 'යන බව', 'යනවා', 'සිටියා', 'සිනහ උණා', 'සෙමෙන්', 'සෙව්වා', 'හම්බුනා', 'හාවා', 'හාවාට', 'හාවෙක්', 'හිතනවට වඩා'

### A few samples from the dataset

https://github.com/user-attachments/assets/216cd975-80d4-4f7d-8a2a-09858f91091a

https://github.com/user-attachments/assets/858b0bfd-d23b-432c-a757-ddf6e58fe70f

dataset password hint is "my guitar brand name & model"

## Data Collection

The `data_collecting.py` script captures actions and processes video frames for each sign symbol using OpenCV and MediaPipe. 
It saves the captured frames and their corresponding keypoints for training. The script supports real-time video capture, guiding the user through the data collection process by displaying prompts and countdowns on the screen.


## Model and Training

A Long Short-Term Memory (LSTM) model was used to identify and classify the sign language symbols. The LSTM model is well-suited for this task because it handles sequential data and captures temporal dependencies.

## Results

The model achieved an accuracy of 90%. Below are the classification report and confusion matrix:

### Classification Report

![Classification Report](https://github.com/user-attachments/assets/062246f4-079a-49da-813c-1d690ff8262e)

### Confusion Matrix

![Confusion Matrix](https://github.com/user-attachments/assets/6e394817-8346-46ef-8c03-f80221be0515)


