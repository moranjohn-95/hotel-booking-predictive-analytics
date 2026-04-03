import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hotel Booking Cancellation Analytics",
    layout="wide",
)

model_results = pd.read_csv("outputs/model_comparison_results.csv")
feature_columns = pd.read_csv(
    "data/processed/X_encoded.csv",
    nrows=0,
).columns.tolist()
cleaned_df = pd.read_csv("data/processed/cleaned_hotel_bookings.csv")
model_path = "models/random_forest_model_deploy.pkl"

model_available = False

try:
    with open(model_path, "rb"):
        model_available = True
except FileNotFoundError:
    model_available = False

model = None

if model_available:
    model = joblib.load(model_path)

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

st.subheader("Hotel type and cancellation rate")

st.write(
    """
    This chart shows that cancellation rates were higher for City Hotel
    bookings than for Resort Hotel bookings, supporting the earlier EDA
    finding that hotel type is related to cancellation behaviour.
    """
)

hotel_cancel_rate = (
    cleaned_df.groupby("hotel")["is_canceled"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

chart_col, _ = st.columns([1, 1])

with chart_col:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    hotel_cancel_rate.plot(kind="bar", ax=ax)
    ax.set_title("Cancellation Rate by Hotel Type", fontsize=12)
    ax.set_xlabel("Hotel Type", fontsize=10)
    ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Deposit type and cancellation rate")

st.write(
    "Non-refund deposits had much higher cancellation rates, "
    "and deposit type was one of the strongest EDA signals "
    "in the project."
)

deposit_cancel_rate = (
    cleaned_df.groupby("deposit_type")["is_canceled"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

chart_col, _ = st.columns([1, 1])

with chart_col:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    deposit_cancel_rate.plot(kind="bar", ax=ax)
    ax.set_title("Cancellation Rate by Deposit Type", fontsize=12)
    ax.set_xlabel("Deposit Type", fontsize=10)
    ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)

st.header("Model comparison")
st.dataframe(model_results)

st.subheader("ROC-AUC comparison across models")

st.write(
    """
    This chart highlights that Random Forest achieved the highest
    ROC-AUC score, which supported its selection as the final model.
    """
)

chart_col, _ = st.columns([1, 1])

with chart_col:
    fig, ax = plt.subplots(figsize=(5, 3))
    model_results.sort_values(
        by="ROC-AUC",
        ascending=False,
    ).plot(
        kind="bar",
        x="Model",
        y="ROC-AUC",
        ax=ax,
        legend=False,
    )
    ax.set_title("Model ROC-AUC Scores", fontsize=12)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("ROC-AUC", fontsize=10)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)

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

st.header("Initial prediction interface")

st.write(
    """
    Enter a small set of booking details below to explore how the
    final model can be used in practice.
    """
)

col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input(
        "Lead time",
        min_value=0,
        max_value=800,
        value=69,
        step=1,
    )
    hotel = st.selectbox(
        "Hotel",
        ["City Hotel", "Resort Hotel"],
    )
    adr = st.number_input(
        "Average daily rate (ADR)",
        min_value=0.0,
        max_value=6000.0,
        value=95.0,
        step=1.0,
    )
    total_of_special_requests = st.number_input(
        "Total special requests",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )
    booking_changes = st.number_input(
        "Booking changes",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )
    previous_cancellations = st.number_input(
        "Previous cancellations",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )

with col2:
    deposit_type = st.selectbox(
        "Deposit type",
        ["No Deposit", "Non Refund", "Refundable"],
    )
    customer_type = st.selectbox(
        "Customer type",
        ["Transient", "Transient-Party", "Contract", "Group"],
    )
    meal = st.selectbox(
        "Meal",
        ["BB", "HB", "SC", "Undefined", "FB"],
    )
    market_segment = st.selectbox(
        "Market segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Groups",
            "Direct",
            "Corporate",
            "Complementary",
            "Aviation",
        ],
    )
    is_repeated_guest = st.selectbox(
        "Is repeated guest?",
        [0, 1],
    )
    previous_bookings_not_canceled = st.number_input(
        "Previous non-cancelled bookings",
        min_value=0,
        max_value=80,
        value=0,
        step=1,
    )

predict_clicked = st.button("Predict cancellation risk")

if predict_clicked:
    input_row = pd.DataFrame(0, index=[0], columns=feature_columns)

    input_row.at[0, "lead_time"] = lead_time
    input_row.at[0, "adr"] = adr
    input_row.at[0, "total_of_special_requests"] = (
        total_of_special_requests
    )
    input_row.at[0, "booking_changes"] = booking_changes
    input_row.at[0, "previous_cancellations"] = previous_cancellations
    input_row.at[0, "previous_bookings_not_canceled"] = (
        previous_bookings_not_canceled
    )
    input_row.at[0, "is_repeated_guest"] = is_repeated_guest

    deposit_col = f"deposit_type_{deposit_type}"
    customer_col = f"customer_type_{customer_type}"
    hotel_col = f"hotel_{hotel}"
    meal_col = f"meal_{meal}"
    market_segment_col = f"market_segment_{market_segment}"

    if deposit_col in input_row.columns:
        input_row.at[0, deposit_col] = 1

    if customer_col in input_row.columns:
        input_row.at[0, customer_col] = 1

    if hotel_col in input_row.columns:
        input_row.at[0, hotel_col] = 1

    if meal_col in input_row.columns:
        input_row.at[0, meal_col] = 1

    if market_segment_col in input_row.columns:
        input_row.at[0, market_segment_col] = 1

    st.subheader("Entered booking details")

    st.write(
        {
            "Lead time": lead_time,
            "ADR": adr,
            "Total special requests": total_of_special_requests,
            "Booking changes": booking_changes,
            "Previous cancellations": previous_cancellations,
            "Previous non-cancelled bookings": (
                previous_bookings_not_canceled
            ),
            "Hotel": hotel,
            "Deposit type": deposit_type,
            "Customer type": customer_type,
            "Meal": meal,
            "Market segment": market_segment,
            "Is repeated guest": is_repeated_guest,
        }
    )

    if model is not None:
        prediction = model.predict(input_row)[0]
        prediction_probability = model.predict_proba(input_row)[0][1]

        if prediction_probability < 0.33:
            risk_band = "Low"
        elif prediction_probability < 0.66:
            risk_band = "Medium"
        else:
            risk_band = "High"

        if risk_band == "High":
            st.error(
                f"Predicted cancellation risk: "
                f"{prediction_probability:.2%}"
            )
        elif risk_band == "Medium":
            st.warning(
                f"Predicted cancellation risk: "
                f"{prediction_probability:.2%}"
            )
        else:
            st.success(
                f"Predicted cancellation risk: "
                f"{prediction_probability:.2%}"
            )

        st.write(f"Risk band: {risk_band}")

        st.caption(
            "This is a model based estimate using the selected "
            "booking inputs. It should be used to support "
            "decision making, not treated as certainty."
        )
    else:
        st.warning(
            "Model file is not available locally, so no prediction "
            "can be generated."
        )

st.subheader("Project sections")
st.write("- Business problem and project overview")
st.write("- Exploratory data analysis findings")
st.write("- Model comparison results")
st.write("- Final Random Forest model")
st.write("- Example prediction interface")
