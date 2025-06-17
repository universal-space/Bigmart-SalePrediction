# main.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# === 1. Load Data ===
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
print("Train shape:", train.shape, "Test shape:", test.shape)

# === 2. Combine Data ===
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)

# === 3. Handle Missing Values ===
data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
data['Item_Visibility'] = data['Item_Visibility'].replace(0, data['Item_Visibility'].median())
data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])

# === 4. Feature Engineering ===
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

# Normalize Fat Content labels
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
})

# === 5. Label Encoding ===
le = LabelEncoder()
categorical_cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 
                    'Outlet_Type', 'Item_Type_Combined', 'Outlet_Identifier']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# === 6. Drop Unused Columns ===
data.drop(columns=['Item_Type', 'Item_Identifier', 'Outlet_Establishment_Year'], inplace=True)

# === 7. Split Back to Train/Test ===
train_final = data[data['source'] == 'train'].drop(columns=['source'])
test_final = data[data['source'] == 'test'].drop(columns=['source', 'Item_Outlet_Sales'])

X = train_final.drop('Item_Outlet_Sales', axis=1)
y = train_final['Item_Outlet_Sales']

# === 8. Train the Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)




# === 9. Predict and Evaluate on Training Set ===
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"Training RMSE: {rmse:.2f}")
print(f"Training R² Score: {r2:.4f}")

# === 10. Save the Model and Feature Order ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, 'models/bigmart_model.pkl')

# Save feature names used during training
feature_order = list(X.columns)
joblib.dump(feature_order, 'models/features.pkl')

print("Model and feature order saved to 'models/'")

# === 11. Predict on Test Set (Optional) ===
test_predictions = model.predict(test_final)
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Test predictions saved to submission.csv")
