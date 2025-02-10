# Track.insights

# Song Popularity Predictor

This project employs machine learning techniques to predict the popularity of songs based on multiple features, such as energy, danceability, and genre. The model is built using historical data from Spotify, analyzing what factors contribute to a song's success.

Prerequisites
Python 3.x
Anaconda (or any Python environment)
Libraries: pandas, numpy, scikit-learn
Usage
Load the Data: The load_data() function retrieves the dataset and ensures it’s loaded and structured correctly.
Clean the Data: The clean_data() function handles missing values and adds new features relevant for modeling.
Build the Model: The run_model() function prepares the features and target variable, splits the data, and trains a Random Forest Regressor using cross-validation and hyperparameter tuning.
Evaluate the Model: The model outputs performance metrics and insights into which features most significantly impact song popularity.
Running the Project
To execute the analysis, run the main script:

# Features Included

Data loading and preprocessing
Handling of missing values
Feature engineering (adding derived features like YearsSinceRelease)
Machine learning model training with Random Forest Regression
Hyperparameter tuning via Grid Search
Evaluation metrics to assess model performance
Feature importance analysis to understand factors driving popularity
Evaluation Metrics
Mean Squared Error (MSE): This metric measures the average squared difference between predicted and actual popularity.
R-squared (R²): Indicates the proportion of variance in the dependent variable (popularity) that can be predicted from the independent variables (features).


# About Dataset
Data was obtained from Kaggle: Best Songs on Spotify for Every Year (2000-2023).

# Improvements Made in the Code
Handling Missing Values: The code efficiently fills missing values by using column means, ensuring that no numerical column has NaN values.
Data Type Conversion: The numeric columns are converted correctly to avoid any type-related errors.
Model Training and Evaluation: The code implements a Random Forest model with hyperparameter tuning, allowing dynamic optimization of model parameters.
