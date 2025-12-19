import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction Using Machine Learning")

@st.cache_data
def load_data():
    return pd.read_csv("Data/diabetes.csv")   # ✅ correct path

data = load_data()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

st.subheader("Enter Medical Details")

preg = st.number_input("Pregnancies", 0)
glu = st.number_input("Glucose Level", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
ins = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("🔴 Person is Diabetic")
    else:
        st.success("🟢 Person is Not Diabetic")

accuracy = accuracy_score(y_test, model.predict(X_test))
st.info(f"Model Accuracy: {accuracy*100:.2f}%")
