import os
import joblib
import pandas as pd

def generate_alerts(threshold=0.5):
    base = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base, 'models', 'rf_model.pkl')
    scaler_path = os.path.join(base, 'models', 'scaler.pkl')
    data_path = os.path.join(base, 'data', 'network_traffic.csv')
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model not found. Train model first.')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    from data_preprocessing import feature_engineer
    X, y = feature_engineer(df)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:,1]
    alerts = df[probs >= threshold].copy()
    alerts['alert_score'] = probs[probs >= threshold]
    alerts_path = os.path.join(base, 'results', 'alerts.csv')
    os.makedirs(os.path.join(base, 'results'), exist_ok=True)
    alerts.to_csv(alerts_path, index=False)
    print(f'Alerts saved to {alerts_path} (count: {len(alerts)})')

if __name__ == '__main__':
    generate_alerts()
