import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_test.pkl")

# Columns expected by the model
model_features = model.feature_names_in_  # sklearn keeps track of training columns

# User inputs
age = st.number_input("Age", min_value=1, max_value=90, value=18)
sex = st.selectbox("Sex", ["F", "M"])
situation = st.selectbox("Situation", ["virgin", "not virgin"])
skin_color = st.selectbox("Skin color", ["white", "yellow", "black", "brown"])
weight = st.number_input("Weight", min_value=20, max_value=100, value=65)
height = st.number_input("Height", min_value=100, max_value=200, value=170)
beauty = st.slider("Beauty on 10", 0, 10, 7)

# Encode categorical variables
sex_num = 0 if sex == "M" else 1
situation_num = 0 if situation == "virgin" else 1
skin_black = 1 if skin_color == "black" else 0
skin_white = 1 if skin_color == "white" else 0
skin_yellow = 1 if skin_color == "yellow" else 0
skin_brown = 1 if skin_color == "brown" else 0

# Prepare input DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex_num],
    "situation": [situation_num],
    "weight": [weight],
    "height": [height],
    "beauty on 10": [beauty],
    "skin_black": [skin_black],
    "skin_brown": [skin_brown],
    "skin_white": [skin_white],
    "skin_yellow": [skin_yellow],
})

# Reorder columns to match model
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing columns with default 0
input_df = input_df[model_features]

# Predict
if st.button("Predict Price"):
    price_pred = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${price_pred:.2f}")
