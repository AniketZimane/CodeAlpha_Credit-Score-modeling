import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    print("🚀 Importing libraries...")

    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv(r"D:\Innovative_things\Code Alpha\ml_model\medical_data.csv")

    if df.empty:
        raise ValueError("Error: The dataset is empty. Check the CSV file path and contents.")

    print("✅ Data loaded successfully!")
    print(df.head())

    # Check for missing values
    print("🔍 Checking missing values before preprocessing...")
    print(df.isnull().sum())

    # Fill missing values (numeric columns only)
    df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

    print("✅ Missing values handled.")

    # Convert categorical to numerical using One-Hot Encoding
    print("🔄 Converting categorical columns to numerical format...")
    categorical_columns = df.select_dtypes(include=['object']).columns
    if "Disease" in categorical_columns:
        categorical_columns = categorical_columns.drop("Disease")  # Exclude target column

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    print("✅ Categorical conversion completed.")

    # Ensure the target variable exists
    if "Disease" not in df.columns:
        raise ValueError("Error: 'Disease' column not found in dataset!")

    # Split into features (X) and target (y)
    print("📊 Splitting dataset into features (X) and target (y)...")
    X = df.drop(columns=["Disease"])
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("✅ Dataset split completed.")

    # Print final feature names
    print("🛠 Feature Names Used for Training:")
    print(list(X.columns))

    # Train model
    print("🛠 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Training complete.")

    # Save trained model
    model_path = r"D:\Innovative_things\Code Alpha\ml_model\disease_model.pkl"
    print(f"💾 Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("✅ Model saved successfully!")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📈 Model Accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"❌ Error occurred: {e}")
