import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'network_traffic.csv')
    return pd.read_csv(path)

def feature_engineer(df):
    df = df.copy()
    # simple features: one-hot encode protocol, normalize ports and counts
    df['src_port_norm'] = df['src_port'] / 65535.0
    df['dst_port_norm'] = df['dst_port'] / 65535.0
    df['packets_per_second'] = df['packet_count'] / (df['flow_duration'].replace(0,1))
    df = pd.concat([df, pd.get_dummies(df['protocol'], prefix='proto')], axis=1)
    features = ['src_port_norm','dst_port_norm','packet_count','byte_count','flow_duration','packets_per_second'] + [c for c in df.columns if c.startswith('proto_')]
    X = df[features].fillna(0)
    y = df['suspicious']
    return X, y

def preprocess_and_split(test_size=0.2, random_state=42):
    df = load_data()
    X, y = feature_engineer(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, scaler
