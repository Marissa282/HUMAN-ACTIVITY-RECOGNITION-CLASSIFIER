# HUMAN-ACTIVITY-RECOGNITION-CLASSIFIER
A machine learning project for classifying human physical activities using motion sensor data. Built with Python and Scikit-learn.

This project implements a supervised machine learning pipeline to classify different types of human physical activity (e.g., walking, standing, laying) based on sensor data. It uses classical ML models like Logistic Regression and Decision Trees to detect activities from time-series features collected via smartphones or wearables.

## Objective

To develop a predictive model that can accurately recognize physical activity patterns from a labeled dataset, improving understanding of behavior through data.

## Models Used

- **Random Forest** (selected for the real time classifier)
- Logistic Regression
- Decision Tree Classifier
- SVM
- LDA
- KNN
- MLP
- NAIVE BAYES
- others 

## Dataset

The dataset includes preprocessed features derived from accelerometer and gyroscope readings. Each row represents a window of sensor data labeled with the corresponding activity.

## Tools & Libraries

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Jupyter Notebook  

## Key Features

- Data preprocessing and feature scaling  
- Exploratory Data Analysis (EDA)  
- Model training, evaluation and comparison  
- Confusion matrix and accuracy reports  
- Activity-wise prediction insights  

## Results

The best performing model achieved an accuracy of approximately **93%**  on the test set, showing effective recognition of the six defined activity classes.


