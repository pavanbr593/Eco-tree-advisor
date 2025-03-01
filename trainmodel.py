import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("treegrowth.csv")

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

# Save the model
joblib.dump(model, "tree_growth_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training complete. Saved as 'tree_growth_model.pkl'.")
