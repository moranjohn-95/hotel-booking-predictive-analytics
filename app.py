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

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Page",
    [
        "Quick Project Summary",
        "EDA Insights",
        "Model Comparison",
        "Prediction Tool",
        "Model Performance",
        "Business Conclusions",
    ],
)

if page == "Quick Project Summary":
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

    st.info(
        "The goal of this project is to predict whether a hotel booking "
        "is likely to be cancelled, using historical booking data.\n\n"
        "The dataset used for this project contains booking information "
        "for both a city hotel and a resort hotel. The workflow includes "
        "exploratory data analysis, data cleaning, feature engineering, "
        "model training, and final model selection."
    )

if page == "EDA Insights":
    st.header("Key EDA findings")

    st.success(
        """
        This page brings together the main findings from the exploratory
        data analysis and highlights the booking features most closely
        associated with cancellation behaviour. The aim is to show which
        patterns stood out most clearly in the dataset before modelling,
        and to support each finding with an optional chart that can be
        expanded when needed.

        **Key patterns identified during EDA included:**

        - **Longer lead times** - were associated with higher cancellation
        risk.
        - **Non-refund deposits** - showed much higher cancellation rates than
          other deposit types.
        - **Repeat guests** - were noticeably less likely to cancel.
        - **Bookings with more special requests** - tended to be less likely
          to cancel.
        - **Hotel type** and **market segment** - also showed meaningful
          differences in cancellation behaviour.
        """
    )

    def format_value(value, decimals=1, suffix=""):
        if pd.isna(value):
            return "n/a"
        return f"{value:.{decimals}f}{suffix}"

    st.subheader("Lead time and cancellation")
    lead_time_by_cancel = (
        cleaned_df.groupby("is_canceled")["lead_time"]
        .mean()
        .reindex([0, 1])
    )
    lead_time_not_cancelled = lead_time_by_cancel.iloc[0]
    lead_time_cancelled = lead_time_by_cancel.iloc[1]
    lead_time_by_cancel.index = ["Not Cancelled", "Cancelled"]
    st.info(
        "Bookings made further in advance tend to have higher "
        "cancellation rates, which suggests uncertainty increases with "
        "longer planning horizons."
    )
    st.write(
        "Cancelled bookings had a noticeably higher average lead time "
        f"({format_value(lead_time_cancelled, 1)} days) than "
        "non-cancelled bookings "
        f"({format_value(lead_time_not_cancelled, 1)} days), suggesting "
        "that reservations made far in advance may be more vulnerable "
        "to changing plans."
    )
    show_lead_time_chart = st.checkbox(
        "Show supporting chart",
        key="lead_time_chart",
    )
    if show_lead_time_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            lead_time_by_cancel.plot(kind="bar", ax=ax)
            ax.set_title(
                "Average Lead Time by Cancellation Status",
                fontsize=12,
            )
            ax.set_xlabel("Cancellation Status", fontsize=10)
            ax.set_ylabel("Average Lead Time (days)", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Deposit type and cancellation")
    deposit_cancel_rate = (
        cleaned_df.groupby("deposit_type")["is_canceled"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    non_refund_rate = deposit_cancel_rate.get("Non Refund", pd.NA)
    no_deposit_rate = deposit_cancel_rate.get("No Deposit", pd.NA)
    refundable_rate = deposit_cancel_rate.get("Refundable", pd.NA)
    st.info(
        "Deposit policy has a strong relationship with cancellations, "
        "especially when bookings are marked as non-refund."
    )
    deposit_text = (
        "Deposit type was one of the clearest signals in the dataset. "
        "Non Refund bookings showed the highest cancellation rate "
        f"({format_value(non_refund_rate, 1, '%')}), compared with "
        f"No Deposit ({format_value(no_deposit_rate, 1, '%')})"
    )
    if not pd.isna(refundable_rate):
        deposit_text += (
            f" and Refundable ({format_value(refundable_rate, 1, '%')})."
        )
    else:
        deposit_text += "."
    st.write(deposit_text)
    show_deposit_chart = st.checkbox(
        "Show supporting chart",
        key="deposit_type_chart",
    )
    if show_deposit_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            deposit_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Deposit Type",
                fontsize=12,
            )
            ax.set_xlabel("Deposit Type", fontsize=10)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Repeat guests and cancellation")
    repeat_cancel_rate = (
        cleaned_df.groupby("is_repeated_guest")["is_canceled"]
        .mean()
        .reindex([0, 1])
        * 100
    )
    repeat_cancel_rate.index = ["Not Repeated", "Repeated"]
    non_repeat_rate = repeat_cancel_rate.iloc[0]
    repeat_rate = repeat_cancel_rate.iloc[1]
    st.info(
        "Repeat guests tend to be more loyal and less likely to cancel, "
        "making this feature a strong behavioural signal."
    )
    st.write(
        "Repeat guests showed much lower cancellation behaviour "
        f"({format_value(repeat_rate, 1, '%')}) than non-repeat guests "
        f"({format_value(non_repeat_rate, 1, '%')}), suggesting that "
        "loyalty and prior relationship with the hotel reduce "
        "cancellation risk."
    )
    show_repeat_guest_chart = st.checkbox(
        "Show supporting chart",
        key="repeat_guests_chart",
    )
    if show_repeat_guest_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            repeat_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Repeat Guest Status",
                fontsize=12,
            )
            ax.set_xlabel("Repeat Guest Status", fontsize=10)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Special requests and cancellation")
    requests_by_cancel = (
        cleaned_df.groupby("is_canceled")["total_of_special_requests"]
        .mean()
        .reindex([0, 1])
    )
    requests_not_cancelled = requests_by_cancel.iloc[0]
    requests_cancelled = requests_by_cancel.iloc[1]
    requests_by_cancel.index = ["Not Cancelled", "Cancelled"]
    st.info(
        "Guests who make more special requests often show stronger "
        "intent to travel, which corresponds with lower cancellations."
    )
    st.write(
        "Non-cancelled bookings averaged "
        f"{format_value(requests_not_cancelled, 1)} special requests "
        f"versus {format_value(requests_cancelled, 1)} for cancelled "
        "bookings, indicating that more requests may signal stronger "
        "travel intent."
    )
    show_special_requests_chart = st.checkbox(
        "Show supporting chart",
        key="special_requests_chart",
    )
    if show_special_requests_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            requests_by_cancel.plot(kind="bar", ax=ax)
            ax.set_title(
                "Average Special Requests by Cancellation Status",
                fontsize=12,
            )
            ax.set_xlabel("Cancellation Status", fontsize=10)
            ax.set_ylabel("Average Special Requests", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Hotel type and cancellation")
    hotel_cancel_rate = (
        cleaned_df.groupby("hotel")["is_canceled"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    city_hotel_rate = hotel_cancel_rate.get("City Hotel", pd.NA)
    resort_hotel_rate = hotel_cancel_rate.get("Resort Hotel", pd.NA)
    st.info(
        "Cancellation rates differ by hotel type, which helps explain "
        "variation in guest behaviour between city and resort stays."
    )
    st.write(
        "City Hotel bookings showed higher cancellation rates "
        f"({format_value(city_hotel_rate, 1, '%')}) than Resort Hotel "
        f"bookings ({format_value(resort_hotel_rate, 1, '%')}), "
        "indicating booking behaviour differs by hotel context."
    )
    show_hotel_chart = st.checkbox(
        "Show supporting chart",
        key="hotel_type_chart",
    )
    if show_hotel_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            hotel_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Hotel Type",
                fontsize=12,
            )
            ax.set_xlabel("Hotel Type", fontsize=10)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Market segment and cancellation")
    market_cancel_rate = (
        cleaned_df.groupby("market_segment")["is_canceled"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    st.info(
        "Cancellation patterns vary across market segments, indicating "
        "differences in booking behaviour by channel."
    )
    st.write(
        "Cancellation rates varied by segment, with Undefined showing "
        "the highest rate, although this category contains very few "
        "observations, and Corporate showing one of the lowest rates at "
        "12.1%. This suggests customer type and booking channel context "
        "can influence risk."
    )
    show_market_segment_chart = st.checkbox(
        "Show supporting chart",
        key="market_segment_chart",
    )
    if show_market_segment_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(5, 3))
            market_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Market Segment",
                fontsize=12,
            )
            ax.set_xlabel("Market Segment", fontsize=10)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=10)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.success(
        "Lead time, deposit type, repeat guest status, special requests, "
        "hotel type, and market segment all showed meaningful "
        "association with cancellation behaviour before modelling. "
        "These findings supported the later predictive modelling stage "
        "by highlighting relevant features and behavioural patterns "
        "useful for prediction."
    )

if page == "Model Comparison":
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

if page == "Prediction Tool":
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

if page == "Model Performance":
    st.header("Model performance")
    st.info(
        "This page will contain confusion matrix, feature importance, "
        "and final model evaluation details."
    )

if page == "Business Conclusions":
    st.header("Business conclusions")
    st.info(
        "This page will contain business implications and final "
        "project conclusions."
    )
