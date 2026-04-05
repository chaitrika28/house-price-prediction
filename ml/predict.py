import joblib
import numpy as np

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_price(features: dict) -> float:
    """
    features keys (California Housing):
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude
    """
    feature_order = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    values = np.array([[features[k] for k in feature_order]])
    values_scaled = scaler.transform(values)
    price = model.predict(values_scaled)[0]
    # NEW - returns Indian Rupees
    USD_TO_INR = 83.5  # Update this rate if needed
    price_in_usd = price * 100000
    price_in_inr = price_in_usd * USD_TO_INR
    return round(price_in_inr, 2)