import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import preprocess_and_split

def train_models():
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split()
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    # SVM (probability can be slow)
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)
    print('SVM Accuracy:', accuracy_score(y_test, svc_pred))
    print(classification_report(y_test, svc_pred))
    # Save best model (here we save RF for demonstration)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(models_dir, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print('Models and scaler saved to models/')

if __name__ == '__main__':
    train_models()
