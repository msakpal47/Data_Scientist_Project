from src.predict import predict_aqi

sample = {
    "CO": 0.6,
    "CO2": 420.0,
    "NO2": 18.0,
    "SO2": 6.0,
    "O3": 30.0,
    "PM2.5": 22.0,
    "PM10": 45.0,
    "Date": "2025-01-01T12:00",
}

print("Pred:", predict_aqi(sample))
