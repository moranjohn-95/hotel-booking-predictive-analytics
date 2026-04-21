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
    "data/processed/X_encoded_deploy.csv",
    nrows=0,
).columns.tolist()
cleaned_df = pd.read_csv("data/processed/cleaned_hotel_bookings.csv")
model_path = "models/gradient_boosting_model_deploy.pkl"

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
        "Project Hypotheses and Validation",
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
        st.success("Final Gradient Boosting model file detected.")
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

if page == "Project Hypotheses and Validation":
    st.header("Project Hypotheses and Validation")
    st.write(
        "This section will outline the key project hypotheses and how they "
        "were validated using the hotel booking data."
    )
    st.subheader(
        "H1: Longer lead times are associated with higher cancellation risk"
    )
    st.info(
        "H1 examines whether bookings made further in advance are more likely "
        "to be cancelled. This hypothesis is important because lead time is "
        "one of the most intuitive booking-behaviour signals in the dataset "
        "and is also one of the strongest features in the final model."
    )
    st.write("**How it was examined**")
    st.write(
        "This hypothesis was examined by comparing lead time patterns across "
        "cancelled and non-cancelled bookings. The analysis focused on "
        "whether cancelled bookings tended to have higher lead times on "
        "average and whether the distribution of lead time differed clearly "
        "between the two outcome groups."
    )
    st.write("**Verdict:** Confirmed")
    st.write(
        "The exploratory analysis supported the view that longer lead times "
        "are associated with higher cancellation risk. This makes practical "
        "sense because bookings made well in advance leave more time for "
        "plans to change, prices to shift, or alternative arrangements to be "
        "made."
    )
    st.write(
        "This finding is important in the context of the project because it "
        "supports the inclusion of lead_time in the final prediction app and "
        "helps explain why it remains one of the most influential variables "
        "in the final Gradient Boosting model."
    )
    lead_time_not_cancelled = cleaned_df.loc[
        cleaned_df["is_canceled"] == 0,
        "lead_time",
    ].dropna()
    lead_time_cancelled = cleaned_df.loc[
        cleaned_df["is_canceled"] == 1,
        "lead_time",
    ].dropna()
    show_h1_chart = st.checkbox("Show H1 supporting chart", key="h1_chart")
    if show_h1_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            ax.boxplot(
                [lead_time_not_cancelled, lead_time_cancelled],
                labels=["Not Cancelled", "Cancelled"],
                patch_artist=True,
            )
            ax.set_title(
                "Lead Time Distribution by Cancellation Status",
                fontsize=11,
            )
            ax.set_xlabel("Booking Outcome", fontsize=9)
            ax.set_ylabel("Lead Time (days)", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
        st.write(
            "The chart supports H1 by showing that cancelled bookings "
            "generally have higher lead times than bookings that were not "
            "cancelled. This reinforces the idea that advance booking horizon "
            "is an important indicator of cancellation risk in this dataset."
        )
    st.subheader(
        "H2: Deposit type is strongly linked to cancellation behaviour"
    )
    st.info(
        "H2 examines whether deposit type is linked to cancellation "
        "behaviour. This hypothesis is important because deposit conditions "
        "reflect booking commitment and are likely to influence whether a "
        "guest follows through with the reservation."
    )
    st.write("**How it was examined**")
    st.write(
        "This hypothesis was examined by comparing cancellation behaviour "
        "across the different deposit categories in the dataset. The "
        "analysis focused on whether cancellation rates differed clearly by "
        "deposit type and whether deposit policy appeared to be a meaningful "
        "booking-risk signal."
    )
    st.write("**Verdict:** Confirmed")
    deposit_cancel_rate = (
        cleaned_df.groupby("deposit_type")["is_canceled"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    non_refund_rate = deposit_cancel_rate.get("Non Refund", pd.NA)
    non_refund_text = (
        "n/a" if pd.isna(non_refund_rate) else f"{non_refund_rate:.1f}%"
    )
    st.write(
        "The exploratory analysis supported the view that deposit type is "
        "strongly linked to cancellation behaviour. In this dataset, Non "
        "Refund bookings showed the highest cancellation rate (94.7%), while "
        "No Deposit (26.7%) and Refundable (24.3%) were lower."
    )
    st.write(
        "This observed pattern is notable because it is more unusual than a "
        "simple assumption that stronger deposit commitment would always be "
        "linked to lower cancellation risk. In the context of the project, "
        "this reinforces the importance of using the actual dataset evidence "
        "rather than relying only on expectation."
    )
    st.write(
        "It also shows that deposit policy is still a meaningful signal when "
        "trying to identify higher-risk bookings, which helps explain why "
        "deposit type remains one of the influential variables in the final "
        "Gradient Boosting model used in the app."
    )
    show_h2_chart = st.checkbox("Show H2 supporting chart", key="h2_chart")
    if show_h2_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            deposit_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title("Cancellation Rate by Deposit Type", fontsize=11)
            ax.set_xlabel("Deposit Type", fontsize=9)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=9)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
        st.write(
            "The chart supports H2 by showing that cancellation behaviour "
            "differs clearly across deposit types, with Non Refund showing "
            f"the highest rate ({non_refund_text}) in this dataset. This "
            "reinforces the idea that deposit conditions act as a strong "
            "indicator of cancellation risk for the project."
        )
    st.subheader(
        "H3: Previous cancellation history increases future cancellation risk"
    )
    st.info(
        "H3 examines whether guests with a history of previous cancellations "
        "are more likely to cancel again. This hypothesis is important "
        "because past booking behaviour may provide a strong signal about "
        "future booking risk."
    )
    st.write("**How it was examined**")
    st.write(
        "This hypothesis was examined by comparing cancellation behaviour "
        "across different levels of previous_cancellations. The analysis "
        "focused on whether bookings linked to guests with more previous "
        "cancellations showed consistently higher cancellation rates."
    )
    st.write("**Verdict:** Confirmed")
    previous_cancel_group = (
        cleaned_df["previous_cancellations"]
        .fillna(0)
        .ge(1)
        .map({False: "0", True: "1+"})
    )
    h3_cancel_rate = (
        cleaned_df.assign(previous_cancel_group=previous_cancel_group)
        .groupby("previous_cancel_group")["is_canceled"]
        .mean()
        .reindex(["0", "1+"])
        * 100
    )
    h3_rate_no_prev = h3_cancel_rate.get("0", pd.NA)
    h3_rate_prev = h3_cancel_rate.get("1+", pd.NA)
    h3_rate_no_prev_text = (
        "n/a" if pd.isna(h3_rate_no_prev) else f"{h3_rate_no_prev:.1f}%"
    )
    h3_rate_prev_text = (
        "n/a" if pd.isna(h3_rate_prev) else f"{h3_rate_prev:.1f}%"
    )
    st.write(
        "The exploratory analysis supported the view that previous "
        "cancellation history increases future cancellation risk. In this "
        "dataset, guests with no previous cancellations had a cancellation "
        f"rate of {h3_rate_no_prev_text}, while guests with one or more "
        f"previous cancellations had a much higher rate of "
        f"{h3_rate_prev_text}."
    )
    st.write(
        "This finding is important in the context of the project because it "
        "shows that the model can benefit from customer-history features, "
        "not just details of the current booking. It also helps explain why "
        "previous_cancellations remains one of the meaningful variables in "
        "the final Gradient Boosting model used in the app."
    )
    show_h3_chart = st.checkbox("Show H3 supporting chart", key="h3_chart")
    if show_h3_chart:
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            h3_cancel_rate.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Previous Cancellations",
                fontsize=11,
            )
            ax.set_xlabel("Previous Cancellations (0 vs 1+)", fontsize=9)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=9)
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
        st.write(
            "The chart supports H3 by showing that cancellation risk rises "
            f"from {h3_rate_no_prev_text} for guests with no previous "
            f"cancellations to {h3_rate_prev_text} for guests with one or "
            "more previous cancellations. This reinforces the idea that past "
            "customer behaviour is an important indicator of future booking "
            "risk in this dataset."
        )
    st.subheader(
        "H4: Repeated guests are less likely to cancel than new guests"
    )
    st.info(
        "H4 examines whether repeated guests are less likely to cancel than "
        "new guests. This hypothesis is important because repeat booking "
        "behaviour may reflect stronger customer commitment and lower "
        "uncertainty around the reservation."
    )
    st.write("**How it was examined**")
    st.write(
        "This hypothesis was examined by comparing cancellation behaviour "
        "between repeated and non-repeated guests. The analysis focused on "
        "whether repeated guests showed lower cancellation rates and more "
        "stable booking behaviour than first-time or non-repeated guests."
    )
    st.write("**Verdict:** Confirmed")
    h4_cancel_rate = (
        cleaned_df.groupby("is_repeated_guest")["is_canceled"]
        .mean()
        .reindex([1, 0])
        * 100
    )
    h4_rate_repeated = h4_cancel_rate.get(1, pd.NA)
    h4_rate_non_repeated = h4_cancel_rate.get(0, pd.NA)
    h4_rate_repeated_text = (
        "n/a" if pd.isna(h4_rate_repeated) else f"{h4_rate_repeated:.1f}%"
    )
    h4_rate_non_repeated_text = (
        "n/a"
        if pd.isna(h4_rate_non_repeated)
        else f"{h4_rate_non_repeated:.1f}%"
    )
    st.write(
        "The exploratory analysis supported the view that repeated guests "
        "are less likely to cancel than non-repeated guests. In this "
        f"dataset, repeated guests had a cancellation rate of "
        f"{h4_rate_repeated_text}, while non-repeated guests had a higher "
        f"rate of {h4_rate_non_repeated_text}."
    )
    st.write(
        "This finding is important in the context of the project because it "
        "shows that customer loyalty and booking history provide useful "
        "signals when identifying cancellation risk. It also helps explain "
        "why repeated guest behaviour remains a meaningful variable in the "
        "final Gradient Boosting model used in the app."
    )
    show_h4_chart = st.checkbox("Show H4 supporting chart", key="h4_chart")
    if show_h4_chart:
        h4_cancel_rate_display = h4_cancel_rate.rename(
            index={1: "Repeated Guests", 0: "Non-Repeated Guests"}
        )
        chart_col, _ = st.columns([1, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(4.8, 3.2))
            h4_cancel_rate_display.plot(kind="bar", ax=ax)
            ax.set_title(
                "Cancellation Rate by Repeat Guest Status",
                fontsize=11,
            )
            ax.set_xlabel("Guest Type", fontsize=9)
            ax.set_ylabel("Cancellation Rate (%)", fontsize=9)
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
        st.write(
            "The chart supports H4 by showing that cancellation risk is lower "
            f"for repeated guests ({h4_rate_repeated_text}) than for "
            f"non-repeated guests ({h4_rate_non_repeated_text}). This "
            "reinforces the idea that prior successful booking behaviour is "
            "an important indicator of lower cancellation risk in this "
            "dataset."
        )
    st.subheader("Conclusion")
    st.info(
        "Overall, the hypothesis testing stage supported the view that hotel "
        "booking cancellation risk is strongly influenced by a small group "
        "of behavioural and booking-related features. Across the four "
        "hypotheses, longer lead times, deposit type, previous cancellation "
        "history, and repeated guest behaviour all showed meaningful "
        "relationships with cancellation outcomes."
    )
    st.write(
        "The results were also substantial in practical terms. Cancelled "
        "bookings showed higher lead times than non-cancelled bookings, Non "
        "Refund bookings had the highest observed cancellation rate at "
        "94.7%, guests with one or more previous cancellations showed a much "
        "higher cancellation rate than those with none (68.0% vs 26.7%), "
        "and repeated guests showed a much lower cancellation rate than "
        "non-repeated guests (7.7% vs 28.3%). Together, these patterns "
        "helped justify why these variables were retained in the final "
        "prediction workflow."
    )
    st.write(
        "This is important because it shows that the final predictive tool "
        "is not based on arbitrary booking inputs. Instead, the selected "
        "features were supported by exploratory analysis, clear behavioural "
        "patterns in the data, and the final model evaluation results. This "
        "gives the tool a stronger analytical foundation and shows a clear "
        "link between business understanding, data exploration, and final "
        "predictive modelling."
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
        project and explains why Gradient Boosting was selected as the
        final model. The comparison is based on multiple evaluation
        metrics so that model selection reflects overall performance
        rather than a single score. Random Forest performed very
        strongly on training data but showed clear overfitting, while
        Gradient Boosting generalised better and was selected as the
        deployment model.

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
        "Gradient Boosting achieved the highest ROC-AUC (0.8080), "
        "followed by Random Forest (0.7848), Logistic Regression "
        "(0.7724), and Decision Tree (0.6716). This indicates the "
        "strongest overall separation between cancelled and non-cancelled "
        "bookings across thresholds."
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
        "Gradient Boosting achieved the strongest F1 Score (0.5526), "
        "with Random Forest close behind (0.5495), then Decision Tree "
        "(0.5133), and Logistic Regression (0.4468). This supports its "
        "overall balance between precision and recall."
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
        "Gradient Boosting was selected as the final model because it "
        "performed better on unseen data and showed more stable "
        "generalisation than Random Forest."
    )
    st.write(
        "Random Forest achieved slightly higher recall (0.506 vs 0.454), "
        "but Gradient Boosting provided the better overall balance and "
        "less overfitting for deployment."
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )
    metric_col1.metric("Accuracy", "0.7975")
    metric_col2.metric("Precision", "0.7054")
    metric_col3.metric("Recall", "0.4542")
    metric_col4.metric("F1 Score", "0.5526")
    metric_col5.metric("ROC-AUC", "0.8080")

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
        the final evaluation context for the selected Gradient Boosting
        model.

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
        - prepares the cleaned, deployment-aligned booking data
        - handles missing values and validates data types
        - encodes the selected categorical booking features into numeric form
        - creates the final feature set used by the model and the prediction
          app

        **Stage 2: Classification Model**
        - applies the trained Gradient Boosting classifier
        - generates a cancellation probability for each booking
        - evaluates performance on unseen test data
        - uses the same feature structure as the app so predictions remain
          consistent at deployment stage
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
            X_all = pd.read_csv("data/processed/X_encoded_deploy.csv")
            y_all = pd.read_csv("data/processed/y_deploy.csv")
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
        strongly to the Gradient Boosting model’s predictions.

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
            st.write(
                "The feature importance chart helps show which booking "
                "variables had the greatest influence on the final Gradient "
                "Boosting model. In the context of this project, this is "
                "useful because it highlights which parts of a booking "
                "appear most strongly linked to cancellation risk."
            )
            st.write(
                "Features such as lead time, deposit type, previous "
                "cancellations, and repeat guest behaviour are especially "
                "meaningful because they relate directly to booking "
                "commitment and past customer patterns. This supports the "
                "wider project aim of identifying higher-risk bookings early "
                "enough for a business to respond with more informed "
                "decision-making."
            )
        else:
            st.info("Feature importance is not available for this model type.")

    st.subheader("Model Performance")
    st.info(
        "The metric summary below compares train and test performance for "
        "the final Gradient Boosting model, helping to assess both "
        "predictive quality and generalisation on unseen data."
    )
    st.write(
        "The train and test results are close across the main metrics, "
        "which suggests that the final Gradient Boosting model "
        "generalises well to unseen data and is not showing the heavy "
        "overfitting seen earlier with Random Forest. The test ROC-AUC of "
        "0.808 and F1-score of 0.553 support the decision to select "
        "Gradient Boosting as the final model, while the more moderate "
        "recall of 0.454 shows that some cancellations are still missed."
    )
    st.write(
        "Overall, this means the final model provides a balanced level of "
        "performance for the project. It is strong enough to support "
        "decision-making around cancellation risk, while still requiring "
        "careful interpretation rather than being treated as a certainty "
        "tool."
    )

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

    st.subheader("Confusion Matrix & Classification Report")
    st.info(
        "This section shows how the final Gradient Boosting model "
        "performed on unseen test data. The confusion matrix compares "
        "the model's predicted outcomes against the actual booking "
        "outcomes, helping to show where the model was correct and "
        "where it made errors."
    )
    st.write(
        "In this project, class 0 represents bookings that were not "
        "cancelled and class 1 represents bookings that were cancelled. "
        "The matrix shows that the model is stronger at identifying "
        "bookings that are likely to go ahead, while its ability to catch "
        "all cancellations is more moderate."
    )
    cm = confusion_matrix(y_test, y_test_pred)
    chart_col, _ = st.columns([1, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(3.8, 2.8))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            ax=ax,
            colorbar=False,
        )
        ax.set_title("Test Set Confusion Matrix", fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    st.caption(
        "The confusion matrix shows how many predictions were correct vs "
        "incorrect, split by each class."
    )

    st.write(
        "This is useful in the context of the project because it shows "
        "the model can separate lower-risk bookings from higher-risk "
        "bookings, but it still misses some cancellations. In other words, "
        "the model is a helpful decision-support tool, but its predictions "
        "should not be treated as certainty."
    )
    st.write(
        "The classification report summarises the model's precision, "
        "recall, and F1-score for each class. In these results, the model "
        "performs more strongly for class 0, with a recall of 0.928, "
        "meaning it is very good at identifying bookings that are likely "
        "to go ahead. For class 1, recall is lower at 0.454, which shows "
        "that the model still misses some cancellations. This supports the "
        "earlier findings that the final model is well balanced overall, "
        "with an accuracy of 0.798, but is still better at recognising "
        "stable bookings than cancelled ones."
    )
    with st.expander("Show classification report"):
        report = classification_report(
            y_test,
            y_test_pred,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        metric_cols = ["precision", "recall", "f1-score"]
        report_df[metric_cols] = report_df[metric_cols].round(3)
        report_df["support"] = report_df["support"].round(0).astype(int)
        st.dataframe(report_df, use_container_width=True)

    st.subheader("Diagnostic Plots")
    st.info(
        "This section provides two additional diagnostic views of the final "
        "Gradient Boosting model using unseen test data. These plots help "
        "show how well the model separates cancelled bookings from bookings "
        "that are likely to go ahead, beyond the headline metrics shown "
        "above."
    )
    st.write(
        "The ROC curve shows the trade-off between the true positive rate "
        "and the false positive rate across different decision thresholds. "
        "The Precision-Recall curve is especially useful here because the "
        "project focuses on identifying cancellation risk, so it helps show "
        "how precision changes as recall increases."
    )
    show_diagnostics = st.checkbox("Show ROC and Precision-Recall plots")
    if show_diagnostics:
        if y_test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            chart_col, _ = st.columns([1, 1])
            with chart_col:
                fig, ax = plt.subplots(figsize=(4.2, 2.9))
                ax.plot(fpr, tpr, label="ROC curve")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_title("ROC Curve", fontsize=11)
                ax.set_xlabel("False Positive Rate", fontsize=9)
                ax.set_ylabel("True Positive Rate", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)

            precision, recall, _ = precision_recall_curve(
                y_test,
                y_test_prob,
            )
            avg_precision = average_precision_score(y_test, y_test_prob)
            with chart_col:
                fig, ax = plt.subplots(figsize=(4.2, 2.9))
                ax.plot(recall, precision)
                ax.set_title(
                    f"Precision-Recall Curve (Average Precision="
                    f"{avg_precision:.3f})",
                    fontsize=11,
                )
                ax.set_xlabel("Recall", fontsize=9)
                ax.set_ylabel("Precision", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)

            st.write(
                "In this project, the ROC curve shows that the final model "
                "has good overall ability to separate cancelled bookings from "
                "non-cancelled bookings, which is reflected in the test "
                "ROC-AUC score of 0.808. This suggests the model can rank "
                "higher-risk and lower-risk bookings reasonably well across "
                "different thresholds."
            )
            st.write(
                "The Precision-Recall curve is also informative because the "
                "project focuses on cancellation risk. The average precision "
                "score of 0.657 shows that the model keeps a useful level of "
                "precision as recall increases, although performance becomes "
                "weaker as it tries to catch more cancelled bookings. Taken "
                "together, these plots support the earlier conclusion that "
                "the final model is well balanced overall, but that there is "
                "still a trade-off between identifying more cancellations and "
                "avoiding too many false alarms."
            )
        else:
            st.warning("Model/data not loaded yet.")

    st.subheader("Business Insights")
    st.success(
        "The final model produces a cancellation-risk probability that can "
        "support business decision-making rather than act as certainty. In a "
        "hotel setting, higher-risk bookings could be flagged for targeted "
        "actions such as reminder messages, deposit requirements, or other "
        "measures aimed at reducing last-minute cancellations."
    )
    st.write(
        "This is useful for the project because it shows how predictive "
        "analytics can move beyond description and support practical action. "
        "Instead of treating all bookings in the same way, a business could "
        "use the model output to focus attention on bookings that appear "
        "more likely to cancel, while still recognising that the prediction "
        "is based on probability rather than guarantee."
    )
    st.write(
        "The evaluation results also showed stable train and test "
        "performance, which supports the reliability of the final Gradient "
        "Boosting model on unseen data. This means the model is suitable as "
        "a decision-support tool within the project, while still requiring "
        "human judgement and awareness of its limitations."
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
