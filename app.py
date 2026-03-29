import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hotel Booking Cancellation Analytics",
    layout="wide",
)

model_results = pd.read_csv("outputs/model_comparison_results.csv")
model_path = "models/random_forest_model.pkl"

model_available = False

try:
    with open(model_path, "rb"):
        model_available = True
except FileNotFoundError:
    model_available = False

st.title("Hotel Booking Cancellation Analytics")

st.write(
    """
    This data app explores hotel booking cancellations and presents
    a predictive analytics workflow based on the Hotel Booking
    Demand dataset.
    """
)

if model_available:
    st.success("Final Random Forest model file detected.")
else:
    st.warning(
        "Final model file not found locally. Prediction features "
        "will be limited until the model file is available."
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

st.header("Key EDA findings")

st.write(
    """
    Exploratory data analysis showed that several features were
    strongly related to cancellation behaviour.
    """
)

st.write(
    """
    Important findings included:
    """
)

st.write("- Longer lead times were associated with more cancellations.")
st.write("- Non refund deposits had extremely high cancellation rates.")
st.write(
    "- Repeat guests were much less likely to cancel than "
    "first time guests."
)
st.write(
    "- Bookings with more special requests were less likely "
    "to cancel."
)
st.write(
    "- Market segment and distribution channel showed clear "
    "differences in cancellation behaviour."
)

st.header("Model comparison")

model_results = pd.read_csv("outputs/model_comparison_results.csv")
st.dataframe(model_results)

st.header("Final model selection")

st.write(
    """
    Random Forest was selected as the final model because
    it achieved the strongest overall performance across
    the main evaluation metrics.
    """
)

st.write("Final Random Forest test performance:")
st.write("- Accuracy: 0.8526")
st.write("- Precision: 0.7787")
st.write("- Recall: 0.6491")
st.write("- F1 Score: 0.7080")
st.write("- ROC-AUC: 0.9108")

st.write(
    """
    Although Gradient Boosting also performed strongly, Random Forest
    produced the best overall balance and the highest ROC-AUC score.
    """
)

st.header("Placeholder- prediction interface")

st.write(
    """
    This section will allow users to enter selected booking details
    and view a cancellation prediction from the final Random Forest
    model.
    """
)

st.info(
    "Interactive prediction inputs will be added in the next step."
)
st.subheader("Project sections")
st.write("- Business problem and project overview")
st.write("- Exploratory data analysis findings")
st.write("- Model comparison results")
st.write("- Final Random Forest model")
st.write("- Example prediction interface")
