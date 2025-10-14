

import pandas as pd
import numpy as np

# Load the preprocessed data (Step 1 of the plan)
try:
    df = pd.read_csv("/content/city_air_weather_osm_preprocessed.csv")
    print("Preprocessed data loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: The file '/content/city_air_weather_osm_preprocessed.csv' was not found.")
    print("Please make sure the preprocessing step was completed successfully and the file exists.")
    df = None # Set df to None to avoid errors in subsequent steps

# Define and apply labeling rules (Steps 2 & 3 of the plan)
if df is not None:
    # Initialize a new column for pollution source labels
    df['pollution_source'] = 'Other' # Default label

    # Define thresholds for 'high' pollution levels (these are examples and may need adjustment)
    # These thresholds should ideally be based on domain knowledge or data distribution analysis
    # Using quantiles on the preprocessed (scaled) data
    threshold_no2 = df['no2'].quantile(0.75) # Example: 75th percentile
    threshold_so2 = df['so2'].quantile(0.75) # Example: 75th percentile
    threshold_pm = df['pm2_5'].quantile(0.75) # Using PM2.5 as a proxy for PM, adjust as needed

    # Apply labeling rules
    # Rule: Close to main road + high NO₂ = Vehicular
    df.loc[(df['road_density'] > df['road_density'].median()) & (df['no2'] > threshold_no2), 'pollution_source'] = 'Vehicular'

    # Rule: Near factory + high SO₂ = Industrial
    df.loc[(df['industrial_density'] > df['industrial_density'].median()) & (df['so2'] > threshold_so2), 'pollution_source'] = 'Industrial'

    # Rule: Near farmland + dry season + high PM = Agricultural
    # Assuming 'dry season' could be inferred from low humidity or high temperature (needs domain knowledge for India)
    # For simplicity here, we'll use a placeholder for 'dry season' or a combination of weather features
    # Let's use low humidity as a simple proxy for dry conditions for this example
    # Using quantile on the preprocessed (scaled) data
    threshold_humidity = df['humidity'].quantile(0.25) # Example: 25th percentile (low humidity)

    df.loc[(df['farmland_density'] > df['farmland_density'].median()) & (df['humidity'] < threshold_humidity) & (df['pm2_5'] > threshold_pm), 'pollution_source'] = 'Agricultural'


    # Display the count of each label
    print("\nPollution Source Label Counts:")
    print(df['pollution_source'].value_counts())

    # Display head of the dataframe with new labels
    print("\nDataFrame with Pollution Source Labels:")
    print(df[['city', 'road_density', 'industrial_density', 'farmland_density', 'no2', 'so2', 'pm2_5', 'humidity', 'pollution_source']].head())

    # You might want to save this labeled dataset for the next steps
    df.to_csv("city_air_weather_osm_labeled.csv", index=False)

    print("\nLabeled dataset saved to 'city_air_weather_osm_labeled.csv'")



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the labeled dataset (Assuming it's saved from the previous step)
try:
    df = pd.read_csv("/content/city_air_weather_osm_labeled.csv")
    print("Labeled data loaded successfully.")
except FileNotFoundError:
    print("Error: The file '/content/city_air_weather_osm_labeled.csv' was not found.")
    print("Please make sure the labeling step was completed successfully and the file exists.")
    df = None

if df is not None:
    # Prepare features and target variable
    features = df.drop(['city', 'lat', 'lon', 'timestamp', 'city_code', 'pollution_source'], axis=1)
    target = df['pollution_source']

    # Handle potential missing values if any remained after initial preprocessing
    features = features.fillna(features.median())

    # Encode target variable
    y_encoded, target_classes = pd.factorize(target)
    y_encoded = pd.Series(y_encoded)

    # Split data into training and testing sets (using the same split as before)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(features, y_encoded, test_size=0.2, random_state=42)
    print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")


    # -----------------------------
    # Hyperparameter Tuning for Random Forest
    # -----------------------------
    print("\nStarting Hyperparameter Tuning for Random Forest...")

    # Define the parameter grid to search
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    # Using a smaller cv value (e.g., 3 or 5) for faster execution during initial testing
    grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                  param_grid=param_grid_rf,
                                  cv=3, # Cross-validation folds
                                  scoring='accuracy', # Metric to optimize
                                  n_jobs=-1) # Use all available cores

    # Perform the grid search on the training data
    grid_search_rf.fit(X_train, y_train_encoded)

    print("\nHyperparameter Tuning for Random Forest Complete.")
    print(f"Best parameters found: {grid_search_rf.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search_rf.best_score_:.4f}")

    # -----------------------------
    # Evaluate the Tuned Random Forest Model
    # -----------------------------
    print("\nEvaluating Tuned Random Forest Model...")

    # Get the best model from the grid search
    best_rf_model = grid_search_rf.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_encoded_rf_tuned = best_rf_model.predict(X_test)

    # Evaluate model performance
    print("\nTuned Random Forest Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded_rf_tuned):.4f}")

    print("\nClassification Report:")
    unique_test_labels = np.unique(y_test_encoded)
    target_names_in_test = [target_classes[i] for i in unique_test_labels]
    print(classification_report(y_test_encoded, y_pred_encoded_rf_tuned, labels=unique_test_labels, target_names=target_names_in_test))

    # Confusion Matrix
    cm_rf_tuned = confusion_matrix(y_test_encoded, y_pred_encoded_rf_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_in_test, yticklabels=target_names_in_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Tuned Random Forest Confusion Matrix')
    plt.show()

!pip install xgboost

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the labeled dataset (Assuming it's saved from the previous step)
try:
    df = pd.read_csv("/content/city_air_weather_osm_labeled.csv")
    print("Labeled data loaded successfully.")
except FileNotFoundError:
    print("Error: The file '/content/city_air_weather_osm_labeled.csv' was not found.")
    print("Please make sure the labeling step was completed successfully and the file exists.")
    df = None

if df is not None:
    # Prepare features and target variable
    features = df.drop(['city', 'lat', 'lon', 'timestamp', 'city_code', 'pollution_source'], axis=1)
    target = df['pollution_source']

    # Handle potential missing values if any remained after initial preprocessing
    features = features.fillna(features.median())

    # Encode target variable
    y_encoded, target_classes = pd.factorize(target)
    y_encoded = pd.Series(y_encoded)

    # Split data into training and testing sets (using the same split as before)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(features, y_encoded, test_size=0.2, random_state=42)
    print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

    # -----------------------------
    # Hyperparameter Tuning for XGBoost
    # -----------------------------
    print("\nStarting Hyperparameter Tuning for XGBoost...")

    # Define the parameter grid to search for XGBoost
    # Note: This is a basic grid, you might need to expand it based on results
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize GridSearchCV for XGBoost
    # Use a smaller cv value (e.g., 3 or 5) for faster execution during initial testing
    grid_search_xgb = GridSearchCV(estimator=XGBClassifier(objective='multi:softprob',
                                                           num_class=len(target_classes),
                                                           random_state=42,
                                                           use_label_encoder=False, # Deprecated in newer versions
                                                           eval_metric='mlogloss'), # Appropriate metric for multi-class
                                   param_grid=param_grid_xgb,
                                   cv=3, # Cross-validation folds
                                   scoring='accuracy', # Metric to optimize
                                   n_jobs=-1) # Use all available cores

    # Perform the grid search on the training data
    grid_search_xgb.fit(X_train, y_train_encoded)

    print("\nHyperparameter Tuning for XGBoost Complete.")
    print(f"Best parameters found: {grid_search_xgb.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search_xgb.best_score_:.4f}")

    # -----------------------------
    # Evaluate the Tuned XGBoost Model
    # -----------------------------
    print("\nEvaluating Tuned XGBoost Model...")

    # Get the best model from the grid search
    best_xgb_model = grid_search_xgb.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_encoded_xgb_tuned = best_xgb_model.predict(X_test)

    # Evaluate model performance
    print("\nTuned XGBoost Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded_xgb_tuned):.4f}")

    print("\nClassification Report:")
    unique_test_labels = np.unique(y_test_encoded)
    target_names_in_test = [target_classes[i] for i in unique_test_labels]
    print(classification_report(y_test_encoded, y_pred_encoded_xgb_tuned, labels=unique_test_labels, target_names=target_names_in_test))

    # Confusion Matrix
    cm_xgb_tuned = confusion_matrix(y_test_encoded, y_pred_encoded_xgb_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_xgb_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_in_test, yticklabels=target_names_in_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Tuned XGBoost Confusion Matrix')
    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the labeled dataset (Assuming it's saved from the previous step)
try:
    df = pd.read_csv("/content/city_air_weather_osm_labeled.csv")
    print("Labeled data loaded successfully.")
except FileNotFoundError:
    print("Error: The file '/content/city_air_weather_osm_labeled.csv' was not found.")
    print("Please make sure the labeling step was completed successfully and the file exists.")
    df = None

if df is not None:
    # Prepare features and target variable
    features = df.drop(['city', 'lat', 'lon', 'timestamp', 'city_code', 'pollution_source'], axis=1)
    target = df['pollution_source']

    # Handle potential missing values if any remained after initial preprocessing
    features = features.fillna(features.median())

    # Encode target variable
    y_encoded, target_classes = pd.factorize(target)
    y_encoded = pd.Series(y_encoded)

    # Split data into training and testing sets (using the same split as before)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(features, y_encoded, test_size=0.2, random_state=42)
    print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

    # -----------------------------
    # Hyperparameter Tuning for Decision Tree
    # -----------------------------
    print("\nStarting Hyperparameter Tuning for Decision Tree...")

    # Define the parameter grid to search for Decision Tree
    # Note: This is a basic grid, you might need to expand it based on results
    param_grid_dt = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Initialize GridSearchCV for Decision Tree
    # Use a smaller cv value (e.g., 3 or 5) for faster execution during initial testing
    grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                                  param_grid=param_grid_dt,
                                  cv=3, # Cross-validation folds
                                  scoring='accuracy', # Metric to optimize
                                  n_jobs=-1) # Use all available cores

    # Perform the grid search on the training data
    grid_search_dt.fit(X_train, y_train_encoded)

    print("\nHyperparameter Tuning for Decision Tree Complete.")
    print(f"Best parameters found: {grid_search_dt.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search_dt.best_score_:.4f}")

    # -----------------------------
    # Evaluate the Tuned Decision Tree Model
    # -----------------------------
    print("\nEvaluating Tuned Decision Tree Model...")

    # Get the best model from the grid search
    best_dt_model = grid_search_dt.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_encoded_dt_tuned = best_dt_model.predict(X_test)

    # Evaluate model performance
    print("\nTuned Decision Tree Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred_encoded_dt_tuned):.4f}")

    print("\nClassification Report:")
    unique_test_labels = np.unique(y_test_encoded)
    target_names_in_test = [target_classes[i] for i in unique_test_labels]
    print(classification_report(y_test_encoded, y_pred_encoded_dt_tuned, labels=unique_test_labels, target_names=target_names_in_test))

    # Confusion Matrix
    cm_dt_tuned = confusion_matrix(y_test_encoded, y_pred_encoded_dt_tuned)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_dt_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_in_test, yticklabels=target_names_in_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Tuned Decision Tree Confusion Matrix')
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
accuracy_dt = 0.9545 # From  (Basic Decision Tree)
accuracy_rf_tuned = 0.9773 # From  (Tuned Random Forest)
accuracy_xgb_tuned = 1.0000 # From  (Tuned XGBoost)

model_names = ['Decision Tree (Basic)', 'Random Forest (Tuned)', 'XGBoost (Tuned)']
accuracies = [accuracy_dt, accuracy_rf_tuned, accuracy_xgb_tuned]

# Create a pandas Series or DataFrame for easy comparison
performance_comparison = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})

# Sort by accuracy for better visualization
performance_comparison = performance_comparison.sort_values(by='Accuracy', ascending=False)

print("Model Performance Comparison (Accuracy):")
display(performance_comparison)

# Optional: Visualize the comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=performance_comparison, palette='viridis')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Comparison of Model Accuracies')
plt.xlim(0, 1.0) # Accuracy is between 0 and 1
plt.show()

# You would typically look at other metrics as well, especially for imbalanced datasets.
# You could manually collect precision, recall, F1-score for each class and compare them here.

print("\nBased on Accuracy, the best performing model is:", performance_comparison.iloc[0]['Model'])

import joblib
import os

# Assuming 'best_xgb_model' is the variable holding your trained XGBoost model
# from the hyperparameter tuning step.
# Make sure to run that cell first to have the best_xgb_model available.

# Define the filename for the exported model
model_filename = 'best_pollution_source_model.joblib'

try:
    # Export the model using joblib
    joblib.dump(best_xgb_model, model_filename)

    print(f"Tuned XGBoost model successfully exported to '{model_filename}'")

    # Verify the file exists
    if os.path.exists(model_filename):
        print(f"File '{model_filename}' created successfully.")
except NameError:
    print("Error: 'best_xgb_model' is not defined.")
    print("Please make sure you have run the XGBoost hyperparameter tuning cell (cell 26331222) successfully.")
except Exception as e:
    print(f"An error occurred during model export: {e}")

