import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #F5F7FA;
}
.title {
    font-size:40px;
    color:#1565C0;
    font-weight:bold;
    text-align:center;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}
.stButton>button{
    background-color:#1565C0;
    color:white;
    border-radius:10px;
    width:100%;
    height:50px;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.markdown('<p class="title">🎓 Student Marks Prediction System</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Predict the final marks of a student using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# Two Columns
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "📚 Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0
    )

    attendance = st.number_input(
        "✅ Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

with col2:
    previous = st.number_input(
        "📝 Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

st.divider()

# Prediction
if st.button("🎯 Predict Final Marks"):

    input_data = pd.DataFrame({
        "Hours_Studied":[hours],
        "Attendance":[attendance],
        "Previous_Marks":[previous]
    })

    prediction = model.predict(input_data)

    st.success(f"🎉 Predicted Final Marks: {prediction[0]:.2f}")
   