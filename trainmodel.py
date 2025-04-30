import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = ["TreeType", "SoilType", "WaterAvailability", "ClimateCondition", "GrowthSuccess"]
        
        # Validate required columns
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
        
        # Validate data types
        if not all(data[col].dtype == 'object' for col in required_columns[:-1]):
            raise ValueError("Categorical columns should be of type object")
        
        if data["GrowthSuccess"].dtype not in ['int64', 'float64']:
            raise ValueError("GrowthSuccess should be numeric")
        
        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_model():
    try:
        # Load and validate data
        data = load_and_validate_data("treegrowth.csv")
        
        # Encode categorical variables
        label_encoders = {}
        for col in ["TreeType", "SoilType", "WaterAvailability", "ClimateCondition"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        
        # Define features and target variable
        X = data[["TreeType", "SoilType", "WaterAvailability", "ClimateCondition"]]
        y = data["GrowthSuccess"]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logging.info(f"Model Accuracy: {accuracy:.2f}")
        logging.info("Classification Report:\n" + report)
        
        # Save the model and encoders
        joblib.dump(model, "tree_growth_model.pkl")
        joblib.dump(label_encoders, "label_encoders.pkl")
        
        logging.info("Model and encoders saved successfully")
        
        return model, label_encoders, accuracy
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, label_encoders, accuracy = train_model()
        print(f"Model training completed successfully with accuracy: {accuracy:.2f}")
    except Exception as e:
        print(f"Error in model training: {str(e)}")
