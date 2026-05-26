import streamlit as st
import requests

st.title("AutoExplainML SaaS")

model = st.file_uploader("Upload Model")
data = st.file_uploader("Upload Data")

if st.button("Run Analysis"):

    res = requests.post(
        "http://localhost:8000/run-job",
        files={
            "model_file": model,
            "data_file": data
        }
    )

    st.json(res.json())