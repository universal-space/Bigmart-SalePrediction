# src/train_model.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    joblib.dump(model, 'models/model.pkl')
    return model
