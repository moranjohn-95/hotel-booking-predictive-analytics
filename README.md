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

## Hypotheses and Validation

The hypotheses for this predictive tool were based on booking and
behavioural patterns that were expected to influence cancellation risk.
These were then checked against the dataset through exploratory
analysis and grouped comparisons before moving into final model
selection.

### Hypothesis Summary

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| **H1:** Longer lead times are associated with higher cancellation risk | Confirmed | Median lead time was **38.0 days** for non cancelled bookings and **80.0 days** for cancelled bookings |
| **H2:** Deposit type is strongly linked to cancellation behaviour | Confirmed | `Non Refund` had the highest cancellation rate at **94.7%**, compared with `No Deposit` at **26.7%** and `Refundable` at **24.3%** |
| **H3:** Previous cancellation history increases future cancellation risk | Confirmed | Guests with no previous cancellations had a cancellation rate of **26.7%**, compared with **68.0%** for guests with one or more previous cancellations |
| **H4:** Repeated guests are less likely to cancel than non repeated guests | Confirmed | Repeated guests had a cancellation rate of **7.7%**, compared with **28.3%** for non repeated guests |

### H1: Longer lead times are associated with higher cancellation risk  

**How it was examined**  
This was examined by comparing lead time patterns across cancelled and
non-cancelled bookings. The analysis focused on whether cancelled
bookings showed higher lead times overall and whether the difference was
clear enough to support inclusion of `lead_time` in the final workflow.

**Verdict**  
Confirmed.

**What the data showed**  
Bookings that were not cancelled had a lower typical lead time
(**median 38.0 days**), while cancelled bookings showed a clearly higher
lead time pattern overall (**median 80.0 days**).

**Interpretation**  
This supported the view that longer lead times are linked to higher
cancellation risk. This makes practical sense because bookings made well
in advance leave more time for plans to change. It also helped justify
why `lead_time` remained one of the most important variables in the
final predictive tool.

### H2: Deposit type is strongly linked to cancellation behaviour.

**How it was examined**  
This was examined by comparing cancellation rates across the deposit
categories in the dataset. The analysis looked at whether cancellation
rates differed clearly by deposit type and whether deposit policy
appeared to be a useful booking-risk signal.

**Verdict**  
Confirmed.

**What the data showed**  
`Non Refund` bookings showed the highest cancellation rate
(**94.7%**), while `No Deposit` (**26.7%**) and `Refundable`
(**24.3%**) were much lower.

**Interpretation**  
This was one of the clearest findings in the dataset, although the
pattern was less straightforward than a simple assumption that stronger
deposit commitment would always lead to lower cancellation risk. This
reinforced the importance of relying on the data itself rather than on
expectation alone. It also helped explain why `deposit_type` remained a
strong feature in the final predictive tool.

### H3: Previous cancellation history increases future cancellation risk

**How it was examined**  
This was examined by comparing cancellation behaviour across different
levels of `previous_cancellations`. The analysis focused on whether
guests with one or more previous cancellations showed consistently
higher cancellation rates than those with none.

**Verdict**  
Confirmed.

**What the data showed**  
Guests with no previous cancellations had a cancellation rate of
**26.7%**, while guests with one or more previous cancellations had a
much higher rate of **68.0%**.

**Interpretation**  
This showed that past customer behaviour can provide a strong signal
about future booking risk. It also supported the inclusion of
`previous_cancellations` in the final workflow and helped show that the
tool benefits not only from current booking details, but also from
customer history.

### H4: Repeated guests are less likely to cancel than non-repeated guests

**How it was examined**  
This was examined by comparing cancellation behaviour between repeated
and non-repeated guests. The analysis focused on whether repeated guests
showed lower cancellation rates and more stable booking behaviour.

**Verdict**  
Confirmed.

**What the data showed**  
Repeated guests had a cancellation rate of **7.7%**, while
non-repeated guests had a higher rate of **28.3%**.

**Interpretation**  
This suggested that customer loyalty and prior successful booking
history are useful signals when identifying cancellation risk. It also
helped justify why repeated guest behaviour remained part of the final
predictive tool.

### Overall Conclusion

The hypothesis testing stage showed that cancellation
risk was strongly influenced by a small group of booking related and
behavioural features.

Longer lead times, deposit type, previous cancellation history, and
repeated guest behaviour all showed meaningful relationships with
cancellation outcomes. These findings helped support the final feature
set and showed that the predictive tool was grounded in clear patterns
from the data.