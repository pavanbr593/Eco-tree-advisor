from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = joblib.load('tree_growth_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tree_type_int = float(request.form['TreeTypeInt'])
        soil_condition = float(request.form['SoilCondition'])
        water_availability = float(request.form['WaterAvailability'])
        climate = float(request.form['Climate'])

        # Prepare the user data for prediction
        user_data = pd.DataFrame({
            "TreeTypeInt": [tree_type_int],
            "SoilCondition": [soil_condition],
            "WaterAvailability": [water_availability],
            "Climate": [climate]
        })

        # Make prediction
        prediction = model.predict(user_data)[0]

        result = "Will Thrive" if prediction == 1 else "Will Not Thrive"

        return jsonify(result=result)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
