import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

# ---------------- CUSTOM CSS (DESIGN) ----------------
st.markdown("""
<style>
body {
    background-color: #FFF7FB;
}
.main {
    background-color: #FFF7FB;
}
.block-container {
    max-width: 800px;
    padding-top: 2rem;
}
h1, h2 {
    text-align: center;
}
.input-card {
    background: white;
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(236, 122, 183, 0.2);
}
.stButton>button {
    background: linear-gradient(135deg, #EC7AB7, #F19AC6);
    color: white;
    border-radius: 999px;
    height: 3rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(236, 122, 183, 0.4);
}
label {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction Using Machine Learning")
st.caption("Predict diabetes risk using medical data and machine learning")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/diabetes.csv")

data = load_data()

# ---------------- PREPARE DATA ----------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ----------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

st.subheader("Enter Medical Details")

preg = st.number_input("Pregnancies", min_value=0)
glu = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
ins = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

predict = st.button("Predict")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict:
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("🔴 High risk of Diabetes detected")
    else:
        st.success("🟢 Low risk of Diabetes detected")
