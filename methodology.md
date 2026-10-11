# Methodology: Telco Customer Churn Prediction

## 1. Research Question
Can we predict whether a telecom customer will cancel their service based on their demographics, subscribed services, contract type, and billing history? Also, how much does our prediction improve when we move from a simple linear model (Logistic Regression) to a tree-based model (Random Forest)?

## 2. Dataset Description
- **Size & Source:** I am using the IBM Telco Customer Churn dataset, which has **7,043 rows** (customers) and **21 columns**.
- **Target Variable:** `Churn` (`Yes` or `No`, converted to `1` and `0`), showing whether the customer left the company within the last month.
- **Features (19 input columns + 1 ID column):**
  - *Customer Info:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - *Account & Billing:* `tenure` (months with the company), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.
  - *Services:* `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.
- **Limitations:**
  1. **Imbalanced Classes:** Only about 26.5% of customers in the dataset actually churned, while 73.5% stayed. This imbalance can trick a standard model into just guessing "No" most of the time.
  2. **Single Snapshot:** The data only shows a customer's status at one point in time rather than tracking their monthly usage changes over time.
  3. **Missing Outside Factors:** We don't have information on competitor prices, network quality, or customer service complaints, which are big reasons people switch providers.

## 3. Data Cleaning Plan
To prepare the raw dataset for modeling, I will do four main cleaning steps:
1. **Remove `customerID`:** This is just a unique ID for each person and does not help predict churn.
2. **Fix `TotalCharges` Data Type:** Right now, `TotalCharges` is stored as text because 11 new customers (where `tenure == 0`) have blank spaces (`" "`) instead of numbers. I will convert this column to numeric (`float`) and fill those 11 blank spots with `0.0`, since new customers haven't been billed yet.
3. **Simplify Category Labels:** Several service columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) have `"No internet service"` as a value, and `MultipleLines` has `"No phone service"`. I will replace these with `"No"` so we don't create unnecessary duplicate columns later.
4. **Convert Target to Numbers:** Map the `Churn` column from `"No"` to `0` and `"Yes"` to `1`.

## 4. Feature Engineering Plan
Before training the models, I will create three new features to help capture customer behavior:
1. **`TenureCohort` (Customer Lifecycle Groups):** Group `tenure` into four buckets: `0-12m_New`, `13-24m_Developing`, `25-48m_Established`, and `49m+_Loyal`. Newer customers are usually much more likely to leave than long-term customers.
2. **`TotalServicesCount`:** Count how many total phone and internet add-on services each customer uses (from 0 to 8). Customers bundled into more services usually find it harder to switch providers.
3. **`ChargeDifference`:** Calculate `MonthlyCharges - (TotalCharges / (tenure + 1))`. This checks if a customer's current monthly bill is noticeably higher than what they usually pay on average.
4. **Encoding & Scaling:** Use One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) to turn text categories into numbers. Then, scale the numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `TotalServicesCount`, `ChargeDifference`) using `StandardScaler`. I will fit the scaler only on the training data to avoid data leakage.

## 5. Models and Why I Chose Them
I will train and compare two classification models using `class_weight='balanced'` to handle the uneven churn numbers:
1. **Logistic Regression (Baseline Model):** I chose this as a starting baseline because it is fast, simple, and easy to interpret. It gives a clear baseline score so we can see if a more complex model is actually worth using.
2. **Random Forest Classifier:** I chose Random Forest as my second model because it combines hundreds of decision trees and handles non-linear patterns well. For example, it can easily pick up combinations like a customer having both a month-to-month contract and expensive fiber optic internet.

## 6. Evaluation Metrics and Why
Because 73.5% of customers in the dataset did not churn, **simple accuracy is not a good metric** (a model that blindly predicts "No Churn" for everyone would still get 73.5% accuracy while missing every actual churner). Instead, I will evaluate the models using:
- **F1-Score (Main Metric):** Balances Precision and Recall. Missing a customer who is about to leave (false negative) costs the company money, while giving discounts to customers who were going to stay anyway (false positive) wastes marketing budget. F1-Score helps balance both mistakes.
- **ROC-AUC:** Measures how well the model separates churners from non-churners across all probability thresholds.
- **Precision & Recall:** I will look at both alongside the confusion matrix to see how many actual churners the model catches (Recall) and how reliable its churn flags are (Precision).