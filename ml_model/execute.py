import pandas as pd

# Load the CSV file
df = pd.read_csv("medical_data.csv")

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())