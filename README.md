# hotel-booking-predictive-analytics
Predictive analytics project using hotel booking demand data, Jupyter notebooks, and Streamlit

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Content](#dataset-content)
3. [Business Requirements](#business-requirements)
4. [Hypotheses and Validation](#hypotheses-and-validation)
5. [Rationale to Map Business Requirements to Data Visualisations and ML Tasks](#rationale-to-map-business-requirements-to-data-visualisations-and-ml-tasks)
6. [ML Business Case](#ml-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Features](#features)
9. [Technologies Used](#technologies-used)
10. [Agile Methodology](#agile-methodology)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [Credits](#credits)
14. [Acknowledgements](#acknowledgements)

## Project Overview

### Purpose

This predictive tool is focused on hotel booking cancellation risk. It
uses historical booking data from the Hotel Booking Demand dataset to
explore which booking patterns are most closely linked to cancellations
and to estimate the likelihood of a booking being cancelled.

The application was built in Streamlit and turns exploratory analysis,
hypothesis testing, model comparison, and final evaluation into a
practical prediction tool. The aim is not to predict cancellation with
certainty, but to provide a structured, evidence-based estimate that can
support booking-risk assessment and decision-making.

### Target Audience

This predictive tool is useful for:

- hotel managers who want better visibility of cancellation risk.
- revenue or operations teams reviewing booking behaviour.
- analysts interested in identifying patterns linked to cancellations.
- businesses exploring how predictive analytics can support booking-risk
  decisions.

### Value Proposition

The value of this predictive tool is that it shows how historical
booking data can be used to move beyond simple reporting and towards
practical risk prediction.

This predictive tool helps to:

- identify booking features linked to higher or lower cancellation risk.
- compare machine learning models in a structured way.
- provide an estimated cancellation-risk score for selected booking
  profiles.
- support more consistent review of bookings that may need closer
  attention.

Overall, the tool demonstrates how predictive analytics can be used to
build a realistic decision support application in a hotel booking
context.

## Dataset Content

The data used for this predictive tool comes from the **Hotel Booking
Demand** dataset, sourced from **Kaggle**.

This dataset contains hotel booking information for two hotel types:

- **City Hotel**
- **Resort Hotel**

It includes historical booking records with details related to booking
timing, customer behaviour, reservation characteristics, and
cancellation outcomes.

### Dataset Summary

- **Dataset name:** Hotel Booking Demand
- **Source:** Kaggle
- **Target variable:** `is_canceled`
- **Prediction type:** Binary classification
- **Outcome classes:**
  - `0` = booking not cancelled
  - `1` = booking cancelled

### Content of the Dataset

The dataset includes a mix of numeric and categorical booking features.
Examples include:

- lead time
- average daily rate (ADR)
- total special requests
- previous cancellations
- previous bookings not cancelled
- repeated guest status
- hotel type
- deposit type
- customer type
- meal plan
- market segment

These features were useful because they describe both the booking itself
and aspects of customer behaviour, which made them suitable for
exploring cancellation risk.

### Why This Dataset Was Suitable

This dataset was suitable for the predictive tool because it contains a
clear cancellation outcome and a wide range of booking related features
that can be analysed before modelling.

It also includes both **City Hotel** and **Resort Hotel** bookings,
which helps provide a broader view of cancellation behaviour across
different hotel contexts.

### Target for the Predictive Tool

The predictive tool was built to estimate whether a booking is likely to
be cancelled. For that reason, the target used throughout the workflow
was:

- `is_canceled`

This made the task a supervised machine learning classification problem,
where the tool learns from past booking outcomes and applies those
patterns to new booking inputs.

### Dataset Considerations

Although the dataset was strong for this type of predictive task, it
still has some limitations.

Some booking fields were not suitable for deployment because they could
introduce leakage or would not be realistically available as app inputs
at prediction time. These fields were removed from the final
deployed workflow so that the predictive tool would stay
realistic and properly match up.

## Business Requirements

The main aim of this predictive tool is to support better understanding
and assessment of hotel booking cancellation risk using historical
booking data.

The business requirements for the tool were defined around 3 main
needs.

### BR1: Understand the booking patterns linked to cancellations

A hotel business needs to understand which booking features are better indicators
and more linked to cancellation behaviour.

This includes identifying patterns such as:

- if longer lead times are linked to higher cancellation risk.
- if deposit type influences cancellation behaviour.
- if repeated guests are less likely to cancel.
- if previous cancellation history affects future risk.
- if hotel type, market segment, and special requests show useful
  behavioural differences.

This requirement was important because a good predictive tool should
not begin with modelling alone. It first needs to show that meaningful
patterns exist in the data.

### BR2: Predict the likelihood of a booking being cancelled

A hotel business needs a tool that can take selected booking inputs and
return an estimated cancellation risk.

This requirement was addressed by building a supervised machine learning
classification workflow that uses historical booking patterns to produce:

- a cancellation risk percentage.
- a risk band.
- a transparent summary of the selected booking profile.

This requirement was important because the aim was not only to analyse
past cancellations, but also to turn that analysis into a working
predictive tool.

### BR3: Support more informed booking-risk decisions

A hotel business needs a practical way to use model output to support
decision making, while recognising that predictions are not certainty.

This means the tool should help users:

- identify bookings that may need closer review.
- interpret the output in a practical way.
- understand the limits of the prediction.
- use the result as decision support rather than as a guaranteed answer.

This requirement was important because the value of the tool depends on
whether the output can be used in a realistic business context.

### How the Predicitve Tool Addresses These Requirements

The finished predictive tool addresses these business requirements by combining:

- exploratory data analysis to identify important patterns.
- hypothesis validation to test key assumptions.
- model comparison to select the strongest final model.
- a prediction interface that returns an estimated cancellation risk.
- evaluation and business conclusions to explain how the output should
  be used in practice.

These requirements helped shape the full workflow and
kept the predicitve tool focused on practical booking risk assessment rather than
prediction for its own sake.