# Car Evaluation ML Classification Model using Random-Forest-Ensemble-Technique
This repository contains a machine learning project that performs classification on the Car Evaluation dataset using the Random Forest ensemble technique.

## Overview
The Car Evaluation dataset is used to classify cars into different categories based on various attributes. This dataset is often used for machine learning and data analysis projects due to its well-structured features and labels.

## Dataset
The Car Evaluation dataset consists of 1,728 instances and includes six attributes (features) and one target variable. The dataset is used to evaluate the quality of cars based on the following features:

### Features
1. **Buying**: The buying price of the car. (Values: vhigh, high, med, low)
2. **Maint**: The maintenance price of the car. (Values: vhigh, high, med, low)
3. **Doors**: The number of doors in the car. (Values: 2, 3, 4, 5more)
4. **Persons**: The capacity of persons to fit in the car. (Values: 2, 4, more)
5. **Lug_boot**: The size of the luggage boot. (Values: small, med, big)
6. **Safety**: The estimated safety of the car. (Values: low, med, high)

### Target Variable
- **Class**: The evaluation of the car. (Values: unacc, acc, good, vgood)

## Machine Learning Algorithm Used
### Random Forest Classifier
Random Forest is an ensemble learning method used for classification, regression, and other tasks. It operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) or mean prediction (regression) of the individual trees.

#### Key Points:
- Combines multiple decision trees to improve the model's performance.
- Reduces overfitting by averaging multiple decision trees.
- Provides feature importance, which helps in understanding the significance of different features.


## Explanation of Code

### Data Preprocessing
- **Loading Data**: The Car Evaluation dataset is loaded using `pandas`.
- **Feature Encoding**: Since the dataset contains categorical features, label encoding is applied to convert categorical values into numerical values using `LabelEncoder` from `sklearn.preprocessing`.
### Model Training
- **Random Forest Classifier**: The Random Forest model is trained using `RandomForestClassifier` from `sklearn.ensemble`.
### Model Evaluation
- **Accuracy**: The model is evaluated based on its accuracy score.
- **Confusion Matrix**: A confusion matrix is generated to visualize the performance of the classifier.
- **Feature Importance**: The importance of each feature is extracted and ranked to understand their significance in the classification process.

## Feature Importance
Random Forest provides a way to evaluate the importance of features in the classification task. By examining the feature importance scores, we can rank the features based on their contribution to the model's decision-making process. In this project, the importance scores of the features were calculated and ranked accordingly.

## Results
- The Random Forest classifier is evaluated based on its accuracy and confusion matrix.
- Feature importance is analyzed to understand the significance of different features in the dataset.
- The ranking of features based on their importance provides insights into which features are most influential in the classification of car evaluations.

## Conclusion
This project demonstrates the use of the Random Forest classifier on the Car Evaluation dataset. The model effectively classifies cars into different categories based on the provided features, with insights on feature importance and model performance.
