import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(): 
    # Step 1: Check your current directory and the contents
    print("Current Working Directory:", os.getcwd())
    data_directory = r'C:\Users\mirun\OneDrive\Documents\GitHub\intro-to-ds-24-25\data'
    print("Directory Contents:", os.listdir(data_directory))

    # Step 2: Attempt to load the CSV file with the correct semicolon separator
    file_path = os.path.join(data_directory, 'Spotify_best_songs2000-2023.csv')  # Correct file path
    

    try:
        df = pd.read_csv(file_path, sep=';')  # Load the CSV file
        print("Dataset Loaded Successfully!")
        print(df.head())  # Check the first few rows of the dataset
        print("Original Column Names:", df.columns.tolist())  # Print original column names
        
        df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace
        print("Corrected Column Names:", df.columns.tolist())  # Check corrected column names
    except FileNotFoundError as e:
        print(e)
        raise SystemExit("Exiting due to file not found. Please check the file path.")

    return df

def clean_data(df):
    # Step 3: Handle Missing Data (fill with mean for numerical)
    df.fillna(df.mean(numeric_only=True), inplace=True) 


    # Convert numerical columns to numeric if they are not already
    numeric_cols = ['energy', 'danceability', 'dB', 'liveness', 'valence', 'duration', 'acousticness', 'speechiness']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, and replace invalid parsing with NaN


    # Step 4: Feature Engineering adding new self-defined columns 
    df['YearsSinceRelease'] = 2023 - df['year']
    df['SongLengthSeconds'] = df['duration'] / 1000  # Assuming 'duration' is in milliseconds

    return df

def run_model(df):
    # Step 5: Prepare Data for Modeling
    X = df.drop(['popularity', 'title', 'artist', 'top genre', 'year'], axis=1)
    Y = df['popularity']

    print("Features (X):")
    print(X.head())
    print("Target (Y):")
    print(Y.head())
    
    # Step 6: Train-test split (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
    # Identify categorical and numerical columns
    categorical_cols = ['top genre']
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    print("Columns After Cleaning: ", df.columns.tolist())
    print("Categorical Columns:", categorical_cols)
    print("Numerical Columns:", numerical_cols)


    # Step 7: Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    

    # Step 8: Define and Train the Model with Hyperparameter Tuning
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Set up the parameter grid for GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }

    
    # Implement Grid Search with Cross-Validation
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, error_score = 'raise')
    

    # Fit the model using the training data
    print("Fitting the model...")
    grid_search.fit(X_train, Y_train)
    
   
    print("Best parameters found: ", grid_search.best_params_)
    
    
    # Step 9: Make Predictions and Evaluate the Best Model
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)  # Make predictions on the test set


    # Step 10: Evaluate the Model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}") 

    # Step 11: Feature Importance (Optional)
    feature_importances = best_model.named_steps['regressor'].feature_importances_
    feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out(numerical_cols).tolist() +
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df) 
    

if __name__== "__main__":
    data = load_data()
    cleaned_data = clean_data(data) 

    print('Building model... ')
    run_model(cleaned_data)

