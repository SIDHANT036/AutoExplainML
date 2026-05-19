import streamlit as st
import pandas as pd
import joblib
from autoexplainml.explainer import explain

st.title("AutoExplainML")

model_file = st.file_uploader("Upload Model (.pkl)")
data_file = st.file_uploader("Upload CSV")

if st.button("Explain"):
    model = joblib.load(model_file)
    data = pd.read_csv(data_file)

    result, _ = explain(model, data)
    st.text(result)