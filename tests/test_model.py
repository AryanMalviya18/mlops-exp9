import joblib, numpy as np

def test_model_loads():
    bundle = joblib.load("model/classifier.pkl")
    assert "model" in bundle and "scaler" in bundle

def test_prediction_shape():
    bundle = joblib.load("model/classifier.pkl")
    X = np.random.rand(1, 30)
    X_scaled = bundle["scaler"].transform(X)
    pred = bundle["model"].predict(X_scaled)
    assert pred.shape == (1,)

def test_prediction_confidence():
    bundle = joblib.load("model/classifier.pkl")
    X = np.random.rand(5, 30)
    X_scaled = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(X_scaled)
    assert proba.shape == (5, 2)