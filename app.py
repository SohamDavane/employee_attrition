# app.py
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "This is a dashboard to predict employee attrition.",
    }
)

# Custom CSS to center text elements
st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
try:
    df = pd.read_csv('realistic_employee_attrition.csv')
except FileNotFoundError:
    st.error("Error: realistic_employee_attrition.csv not found. Please ensure the file is in the correct directory.")
    st.stop()

# Load model
try:
    model = joblib.load("attrition_model_no_overtime_tuned.pkl")
except FileNotFoundError:
    st.error("Error: attrition_model_no_overtime_tuned.pkl not found. Please ensure the model file is in the correct directory.")
    st.stop()

# --- Simulate Evaluation (Replace with your actual evaluation data if available) ---
X_eval = df[['Age', 'Monthly_Salary', 'Years_at_Company', 'Work_Life_Balance', 'Job_Involvement',
             'Hours_Worked_Per_Week', 'Distance_from_Home']]
y_eval = df['Attrition']
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
y_pred_eval = model.predict(X_test_eval)

accuracy = accuracy_score(y_test_eval, y_pred_eval) * 100
precision = precision_score(y_test_eval, y_pred_eval) * 100
recall = recall_score(y_test_eval, y_pred_eval) * 100
f1 = f1_score(y_test_eval, y_pred_eval) * 100
# ----------------------------------------------------------------------------------

# Display the image instead of the main title
st.image('images/ChatGPT Image Apr 11, 2025, 01_29_27 AM.png', caption="Understanding Employee Attrition")
st.markdown("<p class='centered-text'>A tool to predict employee churn.</p>", unsafe_allow_html=True)

st.sidebar.header("Employee Details for Prediction")

with st.sidebar.expander("Basic Information"):
    age = st.number_input("Age", min_value=18, max_value=65, help="Enter the employee's age in years.")

with st.sidebar.expander("Financial Details"):
    salary = st.number_input("Monthly Salary", min_value=1000, max_value=500000, help="Enter the employee's monthly salary.")
    years = st.number_input("Years at Company", min_value=0, max_value=40, help="Enter the number of years the employee has worked at the company.")

with st.sidebar.expander("Work-Life Balance & Involvement"):
    st.markdown("<p style='text-align: center;'>Work Life Balance</p>", unsafe_allow_html=True)
    wlb = st.slider("", 1, 5, 3, format="%d", help="Rate the employee's work-life balance (1=Poor, 5=Excellent).")
    col_wlb_labels = st.columns(5)
    col_wlb_labels[0].markdown("<p style='text-align: left;'>Poor</p>", unsafe_allow_html=True)
    col_wlb_labels[4].markdown("<p style='text-align: right;'>Excellent</p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center;'>Job Involvement</p>", unsafe_allow_html=True)
    involve = st.slider(" ", 1, 5, 3, format="%d", help="Rate the employee's job involvement (1=Low, 5=High).")
    col_involve_labels = st.columns(5)
    col_involve_labels[0].markdown("<p style='text-align: left;'>Low</p>", unsafe_allow_html=True)
    col_involve_labels[4].markdown("<p style='text-align: right;'>High</p>", unsafe_allow_html=True)

with st.sidebar.expander("Work Details"):
    hours = st.number_input("Hours Worked Per Week", min_value=20, max_value=100, value=40, help="Enter the average number of hours the employee works per week.")
    distance = st.number_input("Distance from Home (km)", min_value=1, max_value=100, help="Enter the distance of the employee's home from the company in kilometers.")

# Encode overtime (this line remains for the prediction logic)
overtime_val = 0  # Default value if not provided in the UI

predict_button = st.sidebar.button("Predict", key="predict_button")

col_predict = st.columns(1)[0]  # Use a single column for prediction

if predict_button:
    with col_predict:
        st.subheader("**Prediction Result**")
        input_data = pd.DataFrame({
            'Age': [age],
            'Monthly_Salary': [salary],
            'Years_at_Company': [years],
            'Work_Life_Balance': [wlb],
            'Job_Involvement': [involve],
            'Hours_Worked_Per_Week': [hours],
            'Distance_from_Home': [distance]
        })
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100  # Convert probability to percentage

        if prediction == 1:
            st.error(f"⚠️ Employee likely to leave (Probability: {probability:.2f}%)")
        else:
            st.success(f"✅ Employee likely to stay (Probability: {1 - (probability / 100):.2f}%)") # Corrected stay probability

st.markdown("---")
st.markdown("<h2 class='centered-text'>Overall Employee Retention</h2>", unsafe_allow_html=True)
attrition_rate = df['Attrition'].mean()
stayed_proportion = (1 - attrition_rate) * 100
st.progress(stayed_proportion / 100, text=f"{stayed_proportion:.2f}% of employees retained")

with st.expander("Show Sample Data"):
    st.subheader("First 10 Rows of Dataset")
    st.dataframe(df.head(10))

st.markdown("---")

st.markdown("<h2 class='centered-text'>Feature Importance</h2>", unsafe_allow_html=True)
if hasattr(model, 'feature_importances_'):
    feature_names = ['Age', 'Monthly_Salary', 'Years_at_Company', 'Work_Life_Balance',
                     'Job_Involvement', 'Hours_Worked_Per_Week', 'Distance_from_Home']
    importances = pd.Series(model.feature_importances_, index=feature_names)
    sorted_importances = importances.sort_values(ascending=False)

    fig_importance_line = px.line(sorted_importances, x=sorted_importances.index, y=sorted_importances.values,
                                    title='Feature Importance',
                                    labels={'index': 'Feature', 'y': 'Importance Score'},
                                    markers=True,
                                    line_shape='spline',
                                    color_discrete_sequence=['#66CDAA'],
                                    template='plotly_dark'
                                    )

    fig_importance_line.update_traces(
        marker=dict(size=12,
                    line=dict(width=1.5, color='white')),
        hovertemplate='<b>Feature:</b> %{x}<br><b>Importance:</b> %{y:.3f}<extra></extra>'
    )

    fig_importance_line.update_layout(
        xaxis_title='Feature',
        yaxis_title='Importance Score',
        xaxis=dict(tickangle=-45, automargin=True, showgrid=False, zeroline=False),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11, color='lightgray'),
        title_font=dict(size=16, color='white'),
    )

    st.plotly_chart(fig_importance_line)
else:
    st.warning("Feature importance is not available for the loaded model.")

st.markdown("---")
with st.expander("Model Evaluation"):
    st.write(f"**Accuracy:** {accuracy:.2f}%")
    st.write(f"**Precision (for Attrition=1):** {precision:.2f}%")
    st.write(f"**Recall (for Attrition=1):** {recall:.2f}%")
    st.write(f"**F1-Score (for Attrition=1):** {f1:.2f}%")

st.markdown("---")
st.markdown(f"<p class='centered-text'>📊 Built with Streamlit | Soham Dinesh Davane © | <a href='https://github.com/SohamDavane' target='_blank'>My GitHub</a></p>", unsafe_allow_html=True)
