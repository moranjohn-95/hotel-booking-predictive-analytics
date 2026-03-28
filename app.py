import streamlit as st

st.set_page_config(
    page_title="Hotel Booking Cancellation Analytics",
    layout="wide",
)

st.title("Hotel Booking Cancellation Analytics")

st.write(
    """
    This data app explores hotel booking cancellations and presents
    a predictive analytics workflow based on the Hotel Booking
    Demand dataset.
    """
)

st.header("Project overview")

st.write(
    """
    The goal of this project is to predict whether a hotel booking
    is likely to be cancelled, using historical booking data.
    """
)

st.write(
    """
    The dataset used for this project contains booking information
    for both a city hotel and a resort hotel. The workflow includes
    exploratory data analysis, data cleaning, feature engineering,
    model training, and final model selection.
    """
)

st.subheader("Project sections")
st.write("- Business problem and project overview")
st.write("- Exploratory data analysis findings")
st.write("- Model comparison results")
st.write("- Final Random Forest model")
st.write("- Example prediction interface")
