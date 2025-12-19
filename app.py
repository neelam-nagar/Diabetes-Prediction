import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction Using Machine Learning")
st.write("Enter patient medical details to predict diabetes.")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes.csv")  # PIMA dataset
    return data

data = load_data()

# -------------------------
# Prepare Data
# -------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -------------------------
# User Inputs
# -------------------------
st.subheader("🔍 Enter Medical Information")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🔴 The person is Diabetic")
    else:
        st.success("🟢 The person is Not Diabetic")

# -------------------------
# Model Accuracy
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.info(f"📊 Model Accuracy: {accuracy * 100:.2f}%")