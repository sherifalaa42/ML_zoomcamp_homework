import pickle

from flask import Flask, jsonify
from flask import request

output_file = 'model.bin'

with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()

    X = dv.transform([customer])
    churn = model.predict(X)
    y_pred = model.predict_proba(X)[0, 1]
    

    result = {
        'churn_proba': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)