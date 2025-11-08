import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, 'models', 'rf_model.pkl')
SCALER_PATH = os.path.join(BASE, 'models', 'scaler.pkl')

model = None
scaler = None

def load_artifacts():
    global model, scaler
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    # create required fields
    df['src_port_norm'] = df['src_port'] / 65535.0
    df['dst_port_norm'] = df['dst_port'] / 65535.0
    df['packets_per_second'] = df['packet_count'] / max(1, df['flow_duration'].iloc[0])
    # protocol dummies
    for p in ['TCP','UDP','ICMP']:
        df[f'proto_{p}'] = 1 if df.get('protocol')==p else 0
    features = ['src_port_norm','dst_port_norm','packet_count','byte_count','flow_duration','packets_per_second','proto_TCP','proto_UDP','proto_ICMP']
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    return X_scaled

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_artifacts()
        data = request.get_json()
        X = preprocess_input(data)
        pred = model.predict(X)[0]
        prob = float(model.predict_proba(X)[0,1])
        return jsonify({'prediction': int(pred), 'probability': prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_artifacts()
    app.run(host='0.0.0.0', port=5000)
