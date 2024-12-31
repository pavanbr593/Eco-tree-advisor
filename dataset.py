import pandas as pd
import numpy as np

# Define the conditions
tree_types = [1, 2, 3, 4, 5]  # Example: Areca, Sandalwood, SilverOak, Mango, Coconut
soil_conditions = [1, 2, 3, 4, 5, 6]  # Example: Sandy, Clay, Loam, Peat, Silt, Chalk
water_availability = [1, 2, 3]  # Example: Low, Medium, High
pH_range = (5.0, 8.0)

# Number of records
num_records = 583

# Generate the dataset
def generate_data(num_records):
    # Initialize lists to hold the data
    tree_type = []
    soil_condition = []
    water_availability = []
    pH = []
    outcome = []

    # Generate half of the records with outcome 0 and half with outcome 1
    for i in range(num_records // 2):
        tree_type.append(np.random.choice(tree_types))
        soil_condition.append(np.random.choice(soil_conditions))
        water_availability.append(np.random.choice(water_availability))
        pH.append(round(np.random.uniform(*pH_range), 1))
        outcome.append(0)

    for i in range(num_records // 2):
        tree_type.append(np.random.choice(tree_types))
        soil_condition.append(np.random.choice(soil_conditions))
        water_availability.append(np.random.choice(water_availability))
        pH.append(round(np.random.uniform(*pH_range), 1))
        outcome.append(1)

    # Create a DataFrame
    df = pd.DataFrame({
        'TreeTypeInt': tree_type,
        'SoilCondition': soil_condition,
        'WaterAvailability': water_availability,
        'pH': pH,
        'Outcome': outcome
    })

    return df

# Generate and save the dataset
dataset = generate_data(num_records)
dataset.to_csv('tree_growth_data_with_conditions.csv', index=False)

print(f"Dataset with {num_records} records has been saved as 'tree_growth_data_with_conditions.csv'.")
