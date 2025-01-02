import pandas as pd  # For data manipulation and analysis
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import joblib  # For saving the trained model

# Step 1: Load the dataset
# Assumes a CSV file with feature columns and an "Outcome" column as the target variable
data = pd.read_csv("tree_growth_data_balanced.csv")

# Step 2: Prepare the feature matrix (X) and target vector (Y)
# "Outcome" column is the target variable, all other columns are features
X = data.drop("Outcome", axis=1)  # Drop the "Outcome" column to create the feature matrix
Y = data["Outcome"]  # Target vector

# Step 3: Split the data into training and testing sets
# 80% of the data is used for training, and 20% is used for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Step 4: Initialize and train the Logistic Regression model
model = LogisticRegression()  # Instantiate the logistic regression model
model.fit(X_train, Y_train)  # Fit the model to the training data

# Step 5: Save the trained model to a file
# The model is saved as 'tree_growth_model.pkl' for reuse in prediction scripts
joblib.dump(model, 'tree_growth_model.pkl')

print("Model training complete and saved as 'tree_growth_model.pkl'.")
