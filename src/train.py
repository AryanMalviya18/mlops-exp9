import json, joblib, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("Running train.py...")

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "f1": round(f1_score(y_test, preds), 4),
    "auc": round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 4)
}

print("Metrics:", metrics)

# IMPORTANT: ensure folder exists
import os
os.makedirs("model", exist_ok=True)

joblib.dump({"model": model, "scaler": scaler}, "model/classifier.pkl")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Model + metrics saved")