# Customer Churn Prediction with Logistic Regression

## Business Context
In my experience managing customer retention (Sky Deutschland / WOW), predicting which
customers are likely to cancel is essential for designing proactive retention campaigns.
This project is a simplified version of that process, applying a supervised classification
model to historical customer data.

**Note:** this project uses a public, anonymized dataset randomly selected from an open
database for experimentation purposes — it is not connected to any real business context.
Working with real company data would make this process considerably easier and more
intuitive, since the variables, their business meaning, and the operational levers
available to act on them are already known. The value of this exercise is in demonstrating
the end-to-end methodology, which is directly transferable to a real dataset.

## What I Did
- **Dataset:** Telco Customer Churn (7,043 customers, 21 variables, IBM sample dataset)
- **Preprocessing:** handled missing values in `TotalCharges`, removed non-predictive
  identifiers, one-hot encoded categorical variables, scaled numerical features
- **Model:** Logistic Regression, trained and compared under two configurations
- **Evaluation:** accuracy, precision, recall, and F1-score, with focus on the
  minority class (customers who churned)

## Iterative Process: Two Model Versions

### Version 1 — Baseline model
| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.83 | 0.62 |
| Recall | 0.89 | 0.52 |
| Accuracy (overall) | 79% | |

The baseline model looked strong overall (79% accuracy), but it only caught **52%**
of customers who actually churned — meaning nearly half of at-risk customers would
go undetected by a retention team relying on this model.

### Version 2 — Balanced model (`class_weight='balanced'`)
| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|---|---|---|
| Precision | 0.90 | 0.50 |
| Recall | 0.71 | **0.79** |
| Accuracy (overall) | 73% | |

By rebalancing the model to pay more attention to the minority class, recall for churned
customers jumped from 52% to 79%, at the cost of a lower overall accuracy and more
false positives.

**Business decision:** in a retention context, it is usually cheaper to reach out to
some customers who weren't actually at risk (false positive: an extra email or discount)
than to fail to identify a customer who was genuinely about to leave (false negative:
lost revenue). For that reason, the balanced model — prioritizing recall over precision —
is the more useful choice for a real retention campaign, even though its accuracy is lower.

## Key Churn Drivers
Analyzing the model's coefficients revealed the following top factors:

**Reduce churn risk:**
- **Tenure** (strongest factor overall): the longer a customer has been active, the
  less likely they are to churn — the first few months are the highest-risk period
- **One/two-year contracts**: long-term contracts strongly protect against churn
  compared to month-to-month plans

**Increase churn risk:**
- **Fiber optic internet service**: associated with higher churn, possibly due to
  higher price or stronger competition in that segment
- **Streaming TV / Streaming Movies add-ons**: customers with these services show
  higher churn, possibly reflecting more active value comparison
- **Electronic check as payment method**: associated with higher churn compared to
  automatic payment methods

**Actionable insight:** the three strongest drivers — tenure, contract type, and
internet service type — suggest that the most effective retention actions would be:
(1) strengthening onboarding and engagement in the first few months, when risk is
highest, and (2) incentivizing the migration from month-to-month to annual contracts,
for example through progressive discounts.

## What I Learned
- How to build and evaluate a supervised classification model end-to-end
- Why accuracy alone can be misleading with imbalanced classes, and why recall matters
  more in a retention context
- How to use `class_weight='balanced'` to make a trade-off between precision and recall,
  and how to justify that trade-off from a business perspective
- How to interpret model coefficients to extract actionable business insights, not just
  a prediction score

## Real-World Application
In a real CRM environment, I would connect this type of model to a live customer database,
retrain it regularly on updated data, and feed its output directly into automated retention
flows (e.g. in Braze or mParticle), triggering personalized offers or outreach for customers
flagged as high-risk. Working with real company data would also make it possible to test
additional business-specific variables (support tickets, usage patterns, NPS scores) that
aren't available in this public dataset.

## Tools
Python · pandas · scikit-learn · matplotlib · Google Colab

## How to Run It
1. Open the notebook in Google Colab
2. Run the cells in order (the dataset loads automatically from a public URL)
3. No installation or API keys required
