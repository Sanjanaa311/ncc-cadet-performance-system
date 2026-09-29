
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NCC Cadet Performance Analysis",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("ncc_college_best_model.pkl")
encoder = joblib.load("ncc_college_label_encoder.pkl")
scaler = joblib.load("ncc_college_scaler.pkl")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 10px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 15px;
}

.result-title {
    font-size: 18px;
    font-weight: 600;
}

.result-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

.footer {
    text-align: center;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🪖 NCC Cadet Performance Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Cadet Performance Prediction and '
    'Selection Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "This academic project uses college NCC performance data "
    "to analyse cadet performance and provide a recommendation "
    "for further assessment."
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        "**Project Type:** BCA Mini Project"
    )

    st.write(
        "**Domain:** Machine Learning"
    )

    st.write(
        "**Model:** Logistic Regression"
    )

    st.write(
        "**Dataset:** College NCC Performance Data"
    )

    st.write(
        "**Usable Records:** 30"
    )

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric("Accuracy", "66.67%")
    st.metric("Precision", "75.00%")
    st.metric("Recall", "66.67%")
    st.metric("F1-Score", "66.67%")

    st.divider()

    st.caption(
        "The model is intended for academic demonstration "
        "and decision support."
    )

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Enter Cadet Performance Details</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the assessment values below to generate a performance prediction."
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("🏃 Physical Performance")

    pushup = st.number_input(
        "Pushups",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

    crunches = st.number_input(
        "Crunches",
        min_value=0,
        max_value=100,
        value=25,
        step=1
    )

    situps = st.number_input(
        "Sit-ups",
        min_value=0,
        max_value=100,
        value=25,
        step=1
    )

with col2:

    st.subheader("📚 Profile & Academic Performance")

    height = st.number_input(
        "Height (cm)",
        min_value=100,
        max_value=220,
        value=170,
        step=1
    )

    theory = st.number_input(
        "Theory Marks",
        min_value=0,
        max_value=100,
        value=22,
        step=1
    )

# ============================================================
# INPUT PREVIEW
# ============================================================

input_data = pd.DataFrame({
    "Pushups": [pushup],
    "Crunches": [crunches],
    "Sit-ups": [situps],
    "Height (cm)": [height],
    "Theory Marks": [theory]
})

with st.expander("🔎 View Entered Details"):

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Cadet Performance",
    use_container_width=True
)

if predict_button:

    # Prepare model input
    model_input = pd.DataFrame({
        "PUSHUP": [pushup],
        "CRUNCHES": [crunches],
        "SIT_UPS": [situps],
        "HEIGHT": [height],
        "THEORY": [theory]
    })

    # Scale input
    model_input_scaled = scaler.transform(model_input)

    # Prediction
    prediction = model.predict(
        model_input_scaled
    )[0]

    predicted_category = encoder.inverse_transform(
        [prediction]
    )[0]

    # Probability
    probabilities = model.predict_proba(
        model_input_scaled
    )[0]

    probability_table = pd.DataFrame({
        "Performance Category": encoder.classes_,
        "Probability (%)": probabilities * 100
    })

    probability_table["Probability (%)"] = (
        probability_table["Probability (%)"].round(2)
    )

    # Recommendation
    if predicted_category == "Excellent":

        recommendation = (
            "Recommended for Further Assessment"
        )

        result_message = (
            "The entered performance indicators "
            "correspond to the Excellent category."
        )

    elif predicted_category == "Good":

        recommendation = (
            "Recommended for Further Assessment"
        )

        result_message = (
            "The entered performance indicators "
            "correspond to the Good category."
        )

    elif predicted_category == "Average":

        recommendation = (
            "Requires Further Assessment"
        )

        result_message = (
            "The entered performance indicators "
            "correspond to the Average category."
        )

    else:

        recommendation = (
            "Requires Improvement and Further Assessment"
        )

        result_message = (
            "The entered performance indicators "
            "indicate that further improvement "
            "and assessment may be required."
        )

    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.success(
            f"### Predicted Performance: {predicted_category}"
        )

    with result_col2:

        st.info(
            f"### Recommendation: {recommendation}"
        )

    st.write(result_message)

    # ========================================================
    # PROBABILITY SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Probability</div>',
        unsafe_allow_html=True
    )

    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:

        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True
        )

    with probability_col2:

        chart_data = probability_table.set_index(
            "Performance Category"
        )

        st.bar_chart(
            chart_data
        )

    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.divider()

    st.subheader("🤖 Model Used")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with model_col2:
        st.metric(
            "Training Records",
            "30"
        )

    with model_col3:
        st.metric(
            "Test Accuracy",
            "66.67%"
        )

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project applies machine learning techniques to
        NCC cadet performance data. The system analyses selected
        physical and academic assessment parameters and predicts
        a performance category such as Excellent, Good, Average,
        or Needs Improvement.

        The college dataset contained assessment information
        including pushups, crunches, sit-ups, height and theory
        marks. Since the original OVER ALL field was empty,
        performance categories were derived from the available
        assessment measurements for this academic demonstration.

        Four machine learning algorithms were evaluated:
        Decision Tree, Random Forest, KNN and Logistic Regression.
        Logistic Regression was selected for the final application
        based on the evaluation performed on the available test data.
        """
    )

# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ Decision-Support Notice: This application is an academic "
    "ML project and should not be used as an automatic NCC "
    "selection or rejection system. Final assessment and selection "
    "must follow authorized institutional criteria and human "
    "evaluation."
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'NCC Cadet Performance Analysis System | BCA Mini Project'
    '</div>',
    unsafe_allow_html=True
)
