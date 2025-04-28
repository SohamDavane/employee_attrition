# Employee Attrition Prediction Dashboard

An interactive Streamlit dashboard for predicting employee churn.

This project predicts employee attrition using a machine learning model (Random Forest Classifier) with high accuracy, deployed via a user-friendly web app. It allows users to input employee details, view attrition predictions, and analyze factors affecting employee retention through dynamic visualizations.

---

## Features

- **Churn Prediction:** Real-time employee attrition forecast based on input features.
- **Visual Insights:** Interactive charts showcasing key factors that drive employee attrition.
- **Retention Analysis:** Visualization of overall employee retention and attrition rates.
- **Model Performance:** Trained with high accuracy, precision, recall, and F1-score metrics.

---

## Dashboard Screenshot

![Employee Attrition Dashboard](images/Screenshot%202025-04-28%20112330.png)

---

## Technologies Used

- **Python** — Core development language
- **Pandas** — Data handling and preprocessing
- **Scikit-learn** — Machine Learning model training and evaluation
- **Streamlit** — Web application framework
- **Plotly** — Interactive data visualizations
- **Joblib** — Saving and loading ML models

---

## Dataset

The model was trained using the `realistic_employee_attrition.csv` dataset, which contains features like:

- Age
- Monthly Salary
- Years at Company
- Work-Life Balance
- Job Involvement
- Overtime
- Hours Worked Per Week
- Distance from Home
- Attrition (target variable)

---

## Results

- **Model Accuracy:** ~88%
- **Key Insights:** Work-Life Balance, Overtime, and Job Involvement are among the most significant factors influencing employee attrition.

---
