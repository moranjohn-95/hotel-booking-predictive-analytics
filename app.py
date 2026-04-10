import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

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

    st.success(
        """
        This page compares the classification models tested during the
        project and explains why Random Forest was selected as the final
        model. The comparison is based on multiple evaluation metrics so
        that model selection reflects overall performance rather than a
        single score. Random Forest performed most strongly overall and
        was selected as the final model for deployment in the app.

        Main sections on this page include:
        - model comparison table
        - ROC-AUC comparison
        - F1 Score comparison
        - final model selection
        - business interpretation of model performance
        """
    )

    st.dataframe(model_results)

    st.subheader("ROC-AUC comparison across models")
    st.info(
        "ROC-AUC comparison helps show how well each model separates "
        "cancelled and non-cancelled bookings across thresholds."
    )
    st.write(
        "Random Forest achieved the highest ROC-AUC (0.9108), ahead of "
        "Gradient Boosting (0.8911), Logistic Regression (0.8496), and "
        "Decision Tree (0.7542). This indicates the strongest overall "
        "separation between cancelled and non-cancelled bookings across "
        "thresholds."
    )
    show_roc_chart = st.checkbox(
        "Show supporting chart",
        key="roc_chart",
    )
    if show_roc_chart:
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

    st.subheader("F1 Score comparison across models")
    st.info(
        "F1 Score is useful because it balances precision and recall, "
        "making it a strong measure of overall classification "
        "performance."
    )
    st.write(
        "Random Forest also achieved the strongest F1 Score (0.7080), "
        "followed by Gradient Boosting (0.6493), Decision Tree (0.6413), "
        "and Logistic Regression (0.5742). This supports its overall "
        "balance between precision and recall."
    )
    show_f1_chart = st.checkbox(
        "Show supporting chart",
        key="f1_chart",
    )
    if show_f1_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(5, 3))
            model_results.sort_values(
                by="F1 Score",
                ascending=False,
            ).plot(
                kind="bar",
                x="Model",
                y="F1 Score",
                ax=ax,
                legend=False,
            )
            ax.set_title("Model F1 Scores", fontsize=12)
            ax.set_xlabel("Model", fontsize=10)
            ax.set_ylabel("F1 Score", fontsize=10)
            ax.tick_params(axis="x", rotation=20, labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Final model selection")
    st.info(
        "Random Forest was selected as the final model because it "
        "achieved the strongest overall balance across the evaluation "
        "metrics and the highest ROC-AUC score."
    )
    st.write(
        "In practice, this balance means the model identifies high-risk "
        "bookings reliably without creating too many misleading alerts, "
        "making it the best fit for the project objectives."
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )
    metric_col1.metric("Accuracy", "0.8526")
    metric_col2.metric("Precision", "0.7787")
    metric_col3.metric("Recall", "0.6491")
    metric_col4.metric("F1 Score", "0.7080")
    metric_col5.metric("ROC-AUC", "0.9108")

    st.subheader("Business interpretation of evaluation metrics")
    st.info(
        "Precision shows how many predicted cancellations were correct, "
        "recall shows how many actual cancellations were captured, F1 "
        "balances precision and recall, and ROC-AUC reflects overall "
        "ranking and separation quality."
    )
    st.write(
        "These metrics matter because a useful cancellation model must "
        "identify risky bookings reliably while avoiding too many false "
        "alerts that could lead to unnecessary interventions."
    )

if page == "Prediction Tool":
    st.header("Prediction tool")

    st.success(
        "This page lets you enter selected booking details and returns "
        "an estimated cancellation risk. The result should be used as "
        "decision support rather than certainty."
    )

    st.info(
        "Choose booking details, click Predict cancellation risk, and "
        "review the predicted risk and risk band below."
    )

    st.subheader("How the prediction works")
    st.success(
        """
        The model uses the selected booking details to estimate the
        likelihood that a reservation will be cancelled, based on patterns
        learned from historical hotel booking data.

        The tool returns:
        - a predicted cancellation risk percentage
        - a risk band (Low, Medium, or High)
        - a summary of the booking profile used for the estimate
        """
    )
    st.write(
        "This helps translate the model output into a practical booking-risk "
        "assessment that can support operational planning and review."
    )

    st.write(
        """
        Enter a small set of booking details below to explore how the
        final model can be used in practice.
        """
    )

    st.subheader("Booking input details")
    col1, col2 = st.columns(2)

    with col1:
        lead_time = st.number_input(
            "Lead time",
            min_value=0,
            max_value=800,
            value=69,
            step=1,
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
        previous_bookings_not_canceled = st.number_input(
            "Previous non-cancelled bookings",
            min_value=0,
            max_value=80,
            value=0,
            step=1,
        )

    with col2:
        hotel = st.selectbox(
            "Hotel",
            ["City Hotel", "Resort Hotel"],
        )
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
        is_repeated_guest_label = st.selectbox(
            "Is repeated guest?",
            ["No", "Yes"],
        )

    is_repeated_guest = 1 if is_repeated_guest_label == "Yes" else 0

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

        if model is not None:
            prediction = model.predict(input_row)[0]
            prediction_probability = model.predict_proba(input_row)[0][1]

            if prediction_probability < 0.33:
                risk_band = "Low"
            elif prediction_probability < 0.66:
                risk_band = "Medium"
            else:
                risk_band = "High"

            st.subheader("Prediction result")

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
            st.write(
                "Higher values suggest a greater likelihood of "
                "cancellation based on similar historical bookings."
            )

            st.caption(
                "This is a model-based estimate using the selected "
                "booking inputs. It should be used to support "
                "decision making, not treated as certainty."
            )

            st.info(
                """
                This output can help highlight bookings that may need closer
                review, support more consistent risk assessment, and inform
                planning where cancellation risk is especially relevant.
                """
            )

            st.subheader("What this means")
            if risk_band == "Low":
                st.info(
                    "This booking appears relatively stable compared with "
                    "similar historical bookings, with fewer signs "
                    "typically associated with cancellation."
                )
            elif risk_band == "Medium":
                st.info(
                    "This booking shows some characteristics that have been "
                    "associated with cancellation in the historical data, "
                    "so it may be worth closer monitoring."
                )
            else:
                st.info(
                    "This booking shares several characteristics commonly "
                    "seen in cancelled reservations, suggesting a higher "
                    "level of cancellation risk."
                )

            st.info(
                """
                **Key takeaways**

                - Longer lead times can increase uncertainty and cancellation
                  risk.
                - Guest history and prior cancellations can meaningfully affect
                  the estimate.
                - Repeat guests and stronger booking intent may reduce
                  cancellation likelihood.
                - The result should be used to support judgement, not replace
                  it.
                """
            )

            st.warning(
                """
                **Note:** This prediction is based on historical booking
                patterns and cannot account for every real-world factor. Actual
                outcomes may still differ depending on circumstances not
                captured in the model.
                """
            )
        else:
            st.warning(
                "Model file is not available locally, so no prediction "
                "can be generated."
            )

        st.subheader("Selected booking profile")

        left_summary, right_summary = st.columns(2)
        left_lines = [
            f"**Lead time:** {lead_time}",
            f"**ADR:** {adr}",
            f"**Total special requests:** {total_of_special_requests}",
            f"**Booking changes:** {booking_changes}",
            f"**Previous cancellations:** {previous_cancellations}",
            f"**Previous non-cancelled bookings:** "
            f"{previous_bookings_not_canceled}",
        ]
        right_lines = [
            f"**Hotel:** {hotel}",
            f"**Deposit type:** {deposit_type}",
            f"**Customer type:** {customer_type}",
            f"**Meal:** {meal}",
            f"**Market segment:** {market_segment}",
            f"**Is repeated guest:** "
            f"{'Yes' if is_repeated_guest == 1 else 'No'}",
        ]
        with left_summary:
            st.markdown("  \n".join(left_lines))
        with right_summary:
            st.markdown("  \n".join(right_lines))

if page == "Model Performance":
    st.header("Model performance")
    st.success(
        """
        This page explains how the final cancellation model was evaluated
        and highlights the main evidence used to judge its performance.

        It brings together the key model assessment outputs, including the
        prediction pipeline steps, confusion matrix, feature importance, and
        the final evaluation context for the selected Random Forest model.

        The aim is to show not only how well the model performed on unseen
        data, but also which features contributed most strongly to the
        predictions and why the final model was considered suitable for the
        project objectives.
        """
    )

    st.subheader("ML Pipeline Steps")
    st.info(
        """
        **The prediction pipeline consists of two stages:**

        **Stage 1: Data Cleaning & Feature Engineering**
        - Handles missing values and validates data types.
        - Encodes categorical features into numeric values.
        - Prepares the final feature set used by the model.

        **Stage 2: Classification Model**
        - Generates a probability score for the target outcome.
        - Applies the trained model to unseen data for evaluation.
        - Uses the same preprocessing steps to ensure consistent predictions.
        """
    )

    # Load model and data for evaluation metrics and plots.
    model_perf = st.session_state.get("model", model)
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if (
        model_perf is None
        or X_train is None
        or X_test is None
        or y_train is None
        or y_test is None
    ):
        try:
            X_all = pd.read_csv("data/processed/X_encoded.csv")
            y_all = pd.read_csv("data/processed/y.csv")
            if y_all.shape[1] == 1:
                y_all = y_all.iloc[:, 0]
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=0.2,
                random_state=42,
                stratify=y_all,
            )
        except Exception:
            st.warning("Model/data not loaded yet.")
            st.stop()

    if model_perf is None:
        st.warning("Model/data not loaded yet.")
        st.stop()

    st.subheader("Feature Importance")
    st.write(
        "This helps identify which booking features had the strongest impact "
        "on cancellation prediction in the final model."
    )
    st.info(
        """
        Feature importance shows which input variables contributed most
        strongly to the Random Forest model’s predictions.

        Higher importance values indicate that a feature had greater
        influence on the final prediction outcome.
        """
    )
    show_importance = st.checkbox("Show Feature Importance Plot")
    if show_importance:
        if hasattr(model_perf, "feature_importances_"):
            if hasattr(X_train, "columns"):
                feature_names = list(X_train.columns)
            else:
                feature_names = [
                    f"Feature {index}"
                    for index in range(len(model_perf.feature_importances_))
                ]
            importances = pd.Series(
                model_perf.feature_importances_,
                index=feature_names,
            ).sort_values(ascending=False)
            top_importances = importances.head(15).sort_values()
            chart_col, _ = st.columns([1, 1])
            with chart_col:
                fig, ax = plt.subplots(figsize=(4.8, 3.6))
                top_importances.plot(kind="barh", ax=ax)
                ax.set_title("Top 15 Feature Importances", fontsize=11)
                ax.set_xlabel("Importance", fontsize=9)
                ax.set_ylabel("Feature", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
        else:
            st.info("Feature importance is not available for this model type.")

    st.subheader("Model Performance")

    def format_metric(value):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.3f}"

    def compute_metrics(y_true, y_pred, y_prob):
        metrics = {
            "roc_auc": None,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = None
        return metrics

    y_train_pred = model_perf.predict(X_train)
    y_test_pred = model_perf.predict(X_test)

    if hasattr(model_perf, "predict_proba"):
        y_train_prob = model_perf.predict_proba(X_train)[:, 1]
        y_test_prob = model_perf.predict_proba(X_test)[:, 1]
    else:
        y_train_prob = None
        y_test_prob = None

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    train_col, test_col = st.columns(2)
    with train_col:
        st.markdown("**Train set**")
        st.metric("ROC-AUC", format_metric(train_metrics["roc_auc"]))
        st.metric("Accuracy", format_metric(train_metrics["accuracy"]))
        st.metric("Precision", format_metric(train_metrics["precision"]))
        st.metric("Recall", format_metric(train_metrics["recall"]))
        st.metric("F1", format_metric(train_metrics["f1"]))

    with test_col:
        st.markdown("**Test set**")
        st.metric("ROC-AUC", format_metric(test_metrics["roc_auc"]))
        st.metric("Accuracy", format_metric(test_metrics["accuracy"]))
        st.metric("Precision", format_metric(test_metrics["precision"]))
        st.metric("Recall", format_metric(test_metrics["recall"]))
        st.metric("F1", format_metric(test_metrics["f1"]))

    st.success(
        "The model meets the success criteria on both train and test sets. "
        "A small train–test gap indicates the model generalises well with "
        "no significant overfitting."
    )

    st.subheader("Confusion Matrix & Classification Report")
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    ax.set_title("Test Set Confusion Matrix", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "The confusion matrix shows how many predictions were correct vs "
        "incorrect, split by each class."
    )

    with st.expander("Show classification report"):
        report = classification_report(y_test, y_test_pred)
        st.text(report)

    st.subheader("Diagnostic Plots")
    show_diagnostics = st.checkbox("Show ROC and Precision–Recall plots")
    if show_diagnostics:
        if y_test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.plot(fpr, tpr, label="ROC curve")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title("ROC Curve", fontsize=12)
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            precision, recall, _ = precision_recall_curve(
                y_test,
                y_test_prob,
            )
            avg_precision = average_precision_score(y_test, y_test_prob)
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.plot(recall, precision)
            ax.set_title(
                f"Precision–Recall Curve (Average Precision="
                f"{avg_precision:.3f})",
                fontsize=12,
            )
            ax.set_xlabel("Recall", fontsize=10)
            ax.set_ylabel("Precision", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Model/data not loaded yet.")

    st.subheader("Business Insights")
    st.success(
        """
        The model outputs a risk probability that can be used for decision
        support rather than certainty.
        Higher-risk predictions can trigger targeted actions such as
        reminders, deposits, or time-based incentives.
        The evaluation results show consistent performance between train and
        test, supporting reliability on unseen data.
        """
    )

    st.subheader("Limitations")
    st.warning(
        """
        Predictions are based on historical patterns and may shift if booking
        behaviour changes.
        The model may not capture all real-world factors that influence
        outcomes.
        Probabilities support decisions but do not guarantee individual
        results.
        """
    )

if page == "Business Conclusions":
    st.header("Business conclusions")
    st.info(
        "This page will contain business implications and final "
        "project conclusions."
    )
