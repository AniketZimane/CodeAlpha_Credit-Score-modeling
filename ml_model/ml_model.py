from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
MODEL_PATH = r"D:\Innovative_things\Code Alpha\ml_model\disease_model.pkl"

try:
    print("✅ Loading model...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# Define the expected columns based on the dataset
expected_columns = ['Fever_Yes', 'Cough_Yes', 'Fatigue_Yes', 'DifficultyBreathing_Yes', 
                    'Age', 'Gender_Male', 'BloodPressure_High', 'BloodPressure_Low', 'BloodPressure_Normal',
                    'CholesterolLevel_High', 'CholesterolLevel_Normal']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model is not loaded"}), 500

        data = request.json
        print(f"🔥 Received Data in Flask: {data}")

        # Convert user input into a DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode categorical values
        input_df = pd.get_dummies(input_df)

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure column order matches training data
        input_array = input_df[expected_columns].to_numpy()

        print(f"🧠 Processed Input Data: {input_array}")

        # Make prediction
        prediction = model.predict(input_array)[0]
        print(f"✅ Prediction: {prediction}")

        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
