from flask import Flask, render_template, request, jsonify  # Flask web framework for building the application
import joblib  # For loading pre-trained machine learning models
import pandas as pd  # Data manipulation library
import pickle  # (Optional) For loading serialized objects, though not used here

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model
# Ensure 'tree_growth_model.pkl' exists in the working directory
model = joblib.load('tree_growth_model.pkl')

@app.route('/')
def index():
    """
    Renders the main HTML page of the web application.

    Returns:
        str: Rendered HTML template for the homepage.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction logic for tree growth based on user inputs.

    The function expects the following input fields from the HTML form:
        - TreeTypeInt: Encoded integer representing tree type.
        - SoilCondition: Float value representing soil quality.
        - WaterAvailability: Float value indicating water availability level.
        - Climate: Float value representing climate suitability.

    Returns:
        Response: A JSON object with the prediction result or error details.
    """
    try:
        # Parse input data from the request form
        tree_type_int = float(request.form['TreeTypeInt'])
        soil_condition = float(request.form['SoilCondition'])
        water_availability = float(request.form['WaterAvailability'])
        climate = float(request.form['Climate'])

        # Prepare input data in a format expected by the model
        user_data = pd.DataFrame({
            "TreeTypeInt": [tree_type_int],
            "SoilCondition": [soil_condition],
            "WaterAvailability": [water_availability],
            "Climate": [climate]
        })

        # Perform prediction using the loaded model
        prediction = model.predict(user_data)[0]  # Assumes binary output (1 or 0)

        # Convert the numerical prediction to a human-readable result
        result = "Will Thrive" if prediction == 1 else "Will Not Thrive"

        # Return the result as JSON
        return jsonify(result=result)
    except Exception as e:
        # Handle errors gracefully and return the error message as JSON
        return jsonify(error=str(e)), 400

# Entry point for running the application
if __name__ == '__main__':
    # Run the Flask app in debug mode for development purposes
    app.run(debug=True)
