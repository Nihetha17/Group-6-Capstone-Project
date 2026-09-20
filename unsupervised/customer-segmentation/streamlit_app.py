import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Segmentation")
st.write(
    "Enter the customer's annual income and spending score "
    "to identify the customer segment."
)

annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    value=50.0,
    step=1.0
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)

if st.button("Predict Customer Segment"):

    data = {
        "annual_income_k": annual_income,
        "spending_score": spending_score
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed!")

            st.subheader("Prediction Result")

            st.write("Cluster:", result["cluster"])
            st.write("Customer Segment:", result["persona"])

        else:
            st.error(f"Backend error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure the FastAPI server is running."
        )