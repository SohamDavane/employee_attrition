# 🧑‍💼 Employee Attrition Prediction App

An interactive machine learning web application built using **Streamlit** to predict employee attrition (whether an employee is likely to leave the company) based on key HR metrics.

![App Screenshot](images/Screenshot%202025-04-11%20211741.png)

---

## 📌 About the Project

This project uses a **Random Forest Classifier** trained on a realistic employee dataset to predict attrition.  
The app allows HR managers or analysts to input employee data and instantly get a prediction on whether the employee is at risk of leaving.

The model was trained in a **Google Colab notebook**.

---

## 🧠 How the Model Was Trained

- The dataset used is `realistic_employee_attrition.csv`.
- A `RandomForestClassifier` was trained using features like:
  - Age
  - Monthly Salary
  - Years at Company
  - Work Life Balance
  - Job Involvement
  - Overtime
  - Hours Worked Per Week
  - Distance from Home

> **Note:** The `.pkl` model file is not included in the repo due to GitHub's 25MB limit.  
You can regenerate it using the Colab notebook used during training.

---

## 🚀 Run the App Locally

### 🔧 Prerequisites

Ensure Python is installed. Then install dependencies:

```bash
pip install -r requirements.txt
