import pickle

from flask import Flask, request, jsonify

import pandas as pd

MODEL_FILE = 'model_rf.bin'
threshold = 0.5

with open(MODEL_FILE, 'rb') as f_in:
    # continue
    scaler, model = pickle.load(f_in)

app = Flask('credit_fraud')

@app.route('/predict', methods=['POST'])
def predict():
    transaction = request.get_json()

    # Convert to DataFrame (1-row)
    X = pd.DataFrame([transaction])
    X.columns = X.columns.str.lower()

    # Columns that were scaled during training
    cols_scale = ['time', 'amount']

    # Apply scaler
    X[cols_scale] = scaler.transform(X[cols_scale])

    # Predict fraud probability
    y_pred = model.predict_proba(X)[0, 1]
    fraud = y_pred >= threshold

    result = {
        'fraud_probability': float(y_pred),
        'fraud': bool(fraud)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)