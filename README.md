# Predicting Credit Card Approvals

This repository contains code for analyzing a credit card approval dataset from the UCI Machine Learning Repository. The goal is to predict whether a credit card application will be approved or not.

## Data Preparation
The code imports the necessary libraries, including pandas, and loads the dataset using the read_csv() function. The dataset is read from the file cc_approvals.data with no header information provided. It then inspects the data by printing the first 5 rows, summary statistics, and DataFrame information. The missing values in the dataset are also inspected.

## Data Preprocessing
The code performs various preprocessing steps on the dataset to handle missing values and prepare the data for modeling. It drops two features (columns 11 and 13) that are not relevant for the analysis. The dataset is split into training and testing sets using the train_test_split() function from scikit-learn. The missing values marked as '?' are replaced with NaN. The missing values are then imputed using mean imputation, and the remaining missing values are filled with the most frequent value in each column. Categorical features are converted to dummy variables using one-hot encoding. The columns of the test set are reindexed to align with the train set.

## Feature Scaling
The code uses the MinMaxScaler from scikit-learn to rescale the numerical features in the dataset. The features and labels are segregated into separate variables. The scaler is fitted on the training set and used to transform both the training and testing sets.

## Model Training and Evaluation
A logistic regression model is trained on the rescaled training set using the LogisticRegression class from scikit-learn. The model is then used to predict instances from the test set, and the accuracy score and confusion matrix are calculated and printed.

## Hyperparameter Tuning
GridSearchCV from scikit-learn is used to perform hyperparameter tuning on the logistic regression model. A grid of values for the tol and max_iter parameters is defined, and GridSearchCV is instantiated with the required parameters. The best model is extracted from the grid search results, and its accuracy is evaluated on the test set. The best score and parameters are also printed.

# Notes
- The dataset file cc_approvals.data should be located in the datasets directory within the project folder. Please ensure that you have the dataset file available in the correct location before running the code.
- This code was last updated in September 2021. Please refer to the documentation of the libraries and APIs used for any updates or changes since then.
- Additional preprocessing steps, feature engineering, and model selection can be performed on the dataset to improve the prediction accuracy.
