import streamlit as st
import requests

st.set_page_config(page_title="AutoExplainML", layout="centered")

st.title("🧠 AutoExplainML")
st.write("Upload your ML model and dataset to get explanations")

model_file = st.file_uploader("Upload Model (.pkl)")
data_file = st.file_uploader("Upload Dataset (.csv)")

API_URL = "https://autoexplainml.onrender.com/explain/"

if st.button("Explain Model"):

    if model_file and data_file:

        files = {
            "model_file": model_file,
            "data_file": data_file
        }

        with st.spinner("Analyzing model..."):
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()["explanation"]
            st.success("Done!")

            st.text_area("Explanation", result, height=300)

        else:
            st.error("Error from backend")