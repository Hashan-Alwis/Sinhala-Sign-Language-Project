# Sign Language to Sinhala Text Conversion

## Overview

This project aims to convert sign language symbols into Sinhala text. Each sign language symbol is represented as a sequence of frames in a video. The dataset consists of 26 sign symbols, each with 15 samples.

## Dataset

- **Number of symbols**: 26
- **Samples per symbol**: 15

## Project Structure

- **data_collecting.py**: This script is used for collecting the data.
- **MediaPipe library**: Utilized for processing the video frames.
- **LSTM model**: Employed to identify the sign language symbols.

## Model Performance

The model achieved an accuracy of 90%. Below are the classification report and confusion matrix:

### Classification Report

