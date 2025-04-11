# 🧑‍💼 Employee Attrition Prediction App

An interactive machine learning web application built using **Streamlit** to predict employee attrition (whether an employee is likely to leave the company) based on key HR metrics.

![App Screenshot](images/ChatGPT%20Image%20Apr%2011%2C%202025%2C%2001_29_27%20AM.png)

---

## 📌 About the Project

This project uses a **Random Forest Classifier** trained on a realistic employee dataset to predict attrition.  
The app allows HR managers or analysts to input employee data and instantly get a prediction on whether the employee is at risk of leaving.

The model was trained in **Google Colab**, and the training notebook is included in the repository.

---

## 🧠 How the Model Was Trained

- Training was done in [this Colab Notebook](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/model_training.ipynb)  
  *(📌 Replace the link with your actual Colab URL)*
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

> **Note:** The `.pkl` model file is not included in the repo due to GitHub file size limitations.  
You can regenerate it by running the Colab notebook.

---

## 🚀 Run the App Locally

### 🔧 Prerequisites

Ensure Python is installed. Then install dependencies:

```bash
pip install -r requirements.txt
