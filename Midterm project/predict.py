import pickle

from flask import Flask, request, jsonify

MODEL_FILE = ''
threshold = 0.5

with open(model_file, 'rb') as f_in:
    # continue
    dv, model = pickle.load(f_in)

app = Flask('credit_fraud')

@app.route('/predict', methods=['POST'])
def predict():
    transaction = request.get_json()

    X = dv.transform([transaction])
    y_pred = model.predict_proba(X)[0,1]
    fraud = y_pred >= threshold

    resutl = {
        'fraud_probability': float(y_pred),
        'fraud': bool(fraud)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)