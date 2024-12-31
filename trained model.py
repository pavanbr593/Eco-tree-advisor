import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv("tree_growth_data_balanced.csv")

X = data.drop("Outcome", axis=1)
Y = data["Outcome"]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model
joblib.dump(model, 'tree_growth_model.pkl')
