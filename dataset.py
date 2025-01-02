import pandas as pd  # For handling data frames
import numpy as np  # For generating random data

# Define the conditions for dataset generation
tree_types = [1, 2, 3, 4, 5]  # Example: Encoded values for Areca, Sandalwood, Silver Oak, Mango, Coconut
soil_conditions = [1, 2, 3, 4, 5, 6]  # Example: Encoded values for Sandy, Clay, Loam, Peat, Silt, Chalk
water_availability = [1, 2, 3]  # Example: Encoded values for Low, Medium, High
pH_range = (5.0, 8.0)  # Range for soil pH levels

# Number of records to generate
num_records = 583

def generate_data(num_records):
    """
    Generates a synthetic dataset for tree growth prediction.

    Args:
        num_records (int): The total number of records to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic dataset with the following columns:
            - TreeTypeInt: Encoded tree type.
            - SoilCondition: Encoded soil condition.
            - WaterAvailability: Encoded water availability level.
            - pH: Soil pH level.
            - Outcome: Binary outcome indicating growth success (0 or 1).
    """
    # Initialize lists to hold the data for each column
    tree_type = []
    soil_condition = []
    water_availability_col = []
    pH = []
    outcome = []

    # Generate half the records with outcome 0 (not thriving) and half with outcome 1 (thriving)
    for _ in range(num_records // 2):
        tree_type.append(np.random.choice(tree_types))
        soil_condition.append(np.random.choice(soil_conditions))
        water_availability_col.append(np.random.choice(water_availability))
        pH.append(round(np.random.uniform(*pH_range), 1))
        outcome.append(0)

    for _ in range(num_records // 2):
        tree_type.append(np.random.choice(tree_types))
        soil_condition.append(np.random.choice(soil_conditions))
        water_availability_col.append(np.random.choice(water_availability))
        pH.append(round(np.random.uniform(*pH_range), 1))
        outcome.append(1)

    # Combine the lists into a DataFrame
    df = pd.DataFrame({
        'TreeTypeInt': tree_type,
        'SoilCondition': soil_condition,
        'WaterAvailability': water_availability_col,
        'pH': pH,
        'Outcome': outcome
    })

    return df

# Generate the synthetic dataset
dataset = generate_data(num_records)

# Save the dataset to a CSV file
dataset.to_csv('tree_growth_data_with_conditions.csv', index=False)

# Confirmation message
print(f"Dataset with {num_records} records has been saved as 'tree_growth_data_with_conditions.csv'.")
