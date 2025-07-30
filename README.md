# Heart Disease Prediction using Machine Learning (Logistic Regression)

This project demonstrates how to predict the presence of heart disease in patients using a Machine Learning approach. The Logistic Regression algorithm is implemented using Python to analyze patient data and predict whether heart disease is present.

## Table of Contents
- Overview
- Dataset
- Technologies Used
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Training
- Evaluation
- Results
- How to Run
- License

## Overview
Heart disease is one of the leading causes of death worldwide. Early diagnosis can improve the chances of recovery. In this project, we:
- Perform data analysis and preprocessing
- Build a Logistic Regression model
- Evaluate the model with various metrics
- Predict heart disease based on health indicators

## Dataset
- Dataset: Heart Disease UCI dataset (`heart.csv`)
- Source: [Download Here](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- Features include:
  - Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, and more.
  - Target variable: 1 (disease), 0 (no disease)

## Technologies Used
- Python 3
- Libraries:
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn

## Exploratory Data Analysis (EDA)
- Checked for null values (none found)
- Visualized class distribution (balanced)
- Plotted histograms for categorical and continuous features
- Created correlation heatmap
- Key observations:
  - Chest pain type, thalach, oldpeak show strong correlation with heart disease
  - fbs and chol have low correlation

## Data Preprocessing
- Categorical variables converted to dummy variables using `pd.get_dummies`
- Continuous features scaled using `StandardScaler`
- Data split into 70% training and 30% testing

## Model Training
- Algorithm: Logistic Regression (`solver='liblinear'`)
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix

## Evaluation
Training Results:
- Accuracy: 86.79%
- Confusion Matrix: [[80, 17], [11, 104]]

Testing Results:
- Accuracy: 86.81%
- Confusion Matrix: [[34, 7], [5, 45]]

## Results
The model performs consistently on both training and test sets with an accuracy of around 86.8%. This indicates effective generalization without overfitting.
