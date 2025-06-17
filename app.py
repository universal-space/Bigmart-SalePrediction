# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# === Load trained model and feature names ===
model = joblib.load("models/bigmart_model.pkl")
feature_order = joblib.load("models/features.pkl")  # Load expected column order

# === App UI ===
st.set_page_config(page_title="BigMart Sales Predictor", layout="centered")
st.title("🛍️ BigMart Sales Prediction")
st.markdown("Enter product and outlet information to predict expected sales.")

# === Sidebar Inputs ===
st.sidebar.header("🔧 Product & Outlet Info")

item_weight = st.sidebar.slider("Item Weight (kg)", 0.0, 20.0, 9.3)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
item_mrp = st.sidebar.slider("Item MRP", 30.0, 300.0, 150.0)
outlet_years = st.sidebar.slider("Outlet Age (Years)", 0, 35, 10)

# Categorical selections
fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible'])
location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
size = st.sidebar.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
outlet_type = st.sidebar.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
item_type_combined = st.sidebar.selectbox("Item Type", ['Food', 'Drinks', 'Non-Consumable'])
outlet_id = st.sidebar.selectbox("Outlet ID", ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT035', 'OUT017', 'OUT046', 'OUT019'])

# === Manual Encoding Based on Training ===
label_maps = {
    'Item_Fat_Content': {'Low Fat': 0, 'Non-Edible': 1, 'Regular': 2},
    'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
    'Outlet_Size': {'High': 0, 'Medium': 1, 'Small': 2},
    'Outlet_Type': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3},
    'Item_Type_Combined': {'Drinks': 0, 'Food': 1, 'Non-Consumable': 2},
    'Outlet_Identifier': {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4,
                          'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}
}

# === Build Input DataFrame ===
input_data = {
    'Item_Weight': item_weight,
    'Item_Visibility': item_visibility,
    'Item_MRP': item_mrp,
    'Item_Fat_Content': label_maps['Item_Fat_Content'][fat_content],
    'Outlet_Location_Type': label_maps['Outlet_Location_Type'][location_type],
    'Outlet_Size': label_maps['Outlet_Size'][size],
    'Outlet_Type': label_maps['Outlet_Type'][outlet_type],
    'Item_Type_Combined': label_maps['Item_Type_Combined'][item_type_combined],
    'Outlet_Identifier': label_maps['Outlet_Identifier'][outlet_id],
    'Outlet_Years': outlet_years
}

input_df = pd.DataFrame([input_data])

# === Ensure Column Order Matches Training Data ===
input_df = input_df.reindex(columns=feature_order)

# === Prediction ===
if st.button("🔮 Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Sales: ₹{prediction:.2f}")

# === Feature Importance Visualization ===
st.markdown("---")
st.subheader("🔍 Feature Importance")

importances = model.feature_importances_
features = feature_order

fig, ax = plt.subplots()
ax.barh(features, importances, color='skyblue')
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance from Random Forest Model")
st.pyplot(fig)
