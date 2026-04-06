# ============================================================
# Student Academic Risk Prediction — Streamlit Application
# Author: Your Name
# Description: Interactive web app for predicting student
#              academic risk using a trained LightGBM model
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# page configuration — must be the first streamlit command
# ============================================================
st.set_page_config(
    page_title = "Student Risk Predictor",
    page_icon  = "🎓",
    layout     = "wide"
)

# ============================================================
# loading the saved model and feature names
# ============================================================
@st.cache_resource
def load_model():
    model         = joblib.load('../models/lgbm_final.pkl')
    feature_names = joblib.load('../models/feature_names.pkl')
    scaler        = joblib.load('../models/scaler.pkl')
    return model, feature_names, scaler

model, feature_names, scaler = load_model()

# ============================================================
# loading training data for SHAP explainer background
# ============================================================
@st.cache_resource
def load_explainer():
    df          = pd.read_csv('../data/processed/student_modelling.csv')
    X           = df.drop(columns=['at_risk'])
    explainer   = shap.TreeExplainer(model)
    shap_bg     = X
    return explainer, shap_bg

explainer, shap_bg = load_explainer()

# ============================================================
# app header
# ============================================================
st.title("🎓 Student Academic Risk Prediction System")
st.markdown("""
This tool predicts whether a student is at risk of academic failure 
based on their demographic, socioeconomic, and behavioural profile.

Fill in the student details on the left and click **Predict Risk** 
to generate an instant assessment with explanation.
""")
st.divider()

# ============================================================
# sidebar — student input form
# ============================================================
st.sidebar.title("📋 Student Profile")
st.sidebar.markdown("Enter the student's information below.")

with st.sidebar:

    st.subheader("Demographics")
    age     = st.slider("Age", 15, 22, 17)
    sex     = st.selectbox("Sex", ["M", "F"])
    address = st.selectbox("Address Type", ["U", "R"],
                           help="U = Urban, R = Rural")
    famsize = st.selectbox("Family Size", ["LE3", "GT3"],
                           help="LE3 = ≤3 members, GT3 = >3 members")
    pstatus = st.selectbox("Parent Cohabitation", ["T", "A"],
                           help="T = Together, A = Apart")

    st.subheader("Parental Background")
    medu = st.selectbox("Mother's Education",
                        [0, 1, 2, 3, 4],
                        format_func=lambda x: {
                            0: "0 — None",
                            1: "1 — Primary",
                            2: "2 — Middle School",
                            3: "3 — Secondary",
                            4: "4 — Higher Education"
                        }[x])
    fedu = st.selectbox("Father's Education",
                        [0, 1, 2, 3, 4],
                        format_func=lambda x: {
                            0: "0 — None",
                            1: "1 — Primary",
                            2: "2 — Middle School",
                            3: "3 — Secondary",
                            4: "4 — Higher Education"
                        }[x])
    mjob = st.selectbox("Mother's Job",
                        ["teacher", "health", "services",
                         "at_home", "other"])
    fjob = st.selectbox("Father's Job",
                        ["teacher", "health", "services",
                         "at_home", "other"])

    st.subheader("Academic History")
    failures  = st.slider("Past Class Failures", 0, 4, 0)
    studytime = st.selectbox("Weekly Study Time",
                             [1, 2, 3, 4],
                             format_func=lambda x: {
                                 1: "1 — <2 hours",
                                 2: "2 — 2 to 5 hours",
                                 3: "3 — 5 to 10 hours",
                                 4: "4 — >10 hours"
                             }[x])
    absences  = st.slider("Number of Absences", 0, 93, 4)
    schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
    famsup    = st.selectbox("Family Educational Support", ["yes", "no"])
    paid      = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities= st.selectbox("Extracurricular Activities", ["yes", "no"])
    nursery   = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher    = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet  = st.selectbox("Internet Access at Home", ["yes", "no"])
    romantic  = st.selectbox("In a Romantic Relationship", ["yes", "no"])
    subject   = st.selectbox("Subject", ["math", "portuguese"])

    st.subheader("Lifestyle")
    famrel   = st.slider("Family Relationship Quality", 1, 5, 4,
                         help="1 = Very Bad, 5 = Excellent")
    freetime = st.slider("Free Time After School", 1, 5, 3,
                         help="1 = Very Low, 5 = Very High")
    goout    = st.slider("Going Out with Friends", 1, 5, 3,
                         help="1 = Very Low, 5 = Very High")
    dalc     = st.slider("Workday Alcohol Consumption", 1, 5, 1,
                         help="1 = Very Low, 5 = Very High")
    walc     = st.slider("Weekend Alcohol Consumption", 1, 5, 1,
                         help="1 = Very Low, 5 = Very High")
    health   = st.slider("Current Health Status", 1, 5, 4,
                         help="1 = Very Bad, 5 = Very Good")
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])
    reason   = st.selectbox("Reason for Choosing School",
                            ["home", "reputation",
                             "course", "other"])
    traveltime = st.selectbox("Travel Time to School",
                              [1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "1 — <15 min",
                                  2: "2 — 15 to 30 min",
                                  3: "3 — 30 min to 1 hr",
                                  4: "4 — >1 hr"
                              }[x])

    predict_btn = st.button("🔍 Predict Risk", type="primary",
                            use_container_width=True)

# ============================================================
# feature engineering — replicating Phase 6 transformations
# on the single input student
# ============================================================
def engineer_features(raw_input):
    d = raw_input.copy()

    # replicating all engineered features from notebook 04
    famsup_bin    = 1 if d['famsup']    == 'yes' else 0
    schoolsup_bin = 1 if d['schoolsup'] == 'yes' else 0
    paid_bin      = 1 if d['paid']      == 'yes' else 0

    d['avg_alcohol']               = (d['Dalc'] + d['Walc']) / 2
    d['parent_edu_avg']            = (d['Medu'] + d['Fedu']) / 2
    d['parent_edu_max']            = max(d['Medu'], d['Fedu'])
    d['total_support']             = famsup_bin + schoolsup_bin + paid_bin
    d['social_score']              = d['famrel'] + d['freetime'] + d['goout']
    d['health_risk_score']         = (d['avg_alcohol'] +
                                      (6 - d['health']) +
                                      np.log1p(d['absences']))
    d['is_first_gen']              = 1 if (d['Medu'] < 3 and d['Fedu'] < 3) else 0
    d['failure_absence_interaction']= d['failures'] * np.log1p(d['absences'])
    d['study_support_ratio']       = d['studytime'] / (d['total_support'] + 1)

    return d

# ============================================================
# build input dataframe — matching training feature structure
# ============================================================
def build_input_df(raw):
    engineered = engineer_features(raw)

    # building a single row dataframe
    df_input = pd.DataFrame([engineered])

    # encoding binary columns
    binary_map = {
        'schoolsup': {'yes': 1, 'no': 0},
        'famsup':    {'yes': 1, 'no': 0},
        'paid':      {'yes': 1, 'no': 0},
        'activities':{'yes': 1, 'no': 0},
        'nursery':   {'yes': 1, 'no': 0},
        'higher':    {'yes': 1, 'no': 0},
        'internet':  {'yes': 1, 'no': 0},
        'romantic':  {'yes': 1, 'no': 0},
        'sex':       {'M': 1, 'F': 0},
        'address':   {'U': 1, 'R': 0},
        'famsize':   {'GT3': 1, 'LE3': 0},
        'Pstatus':   {'T': 1, 'A': 0},
    }
    for col, mapping in binary_map.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].map(mapping)

    # one-hot encoding multi-class columns
    multiclass_cols = ['Mjob', 'Fjob', 'reason', 'guardian', 'subject']
    df_input = pd.get_dummies(df_input, columns=multiclass_cols)

    # aligning with training feature names — adding missing columns as 0
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # keeping only training columns in correct order
    df_input = df_input[feature_names]

    return df_input

# ============================================================
# prediction and display
# ============================================================
if predict_btn:

    # assembling raw input dictionary
    raw_input = {
        'age': age, 'sex': sex, 'address': address,
        'famsize': famsize, 'Pstatus': pstatus,
        'Medu': medu, 'Fedu': fedu,
        'Mjob': mjob, 'Fjob': fjob,
        'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup,
        'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery,
        'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': famrel,
        'freetime': freetime, 'goout': goout,
        'Dalc': dalc, 'Walc': walc,
        'health': health, 'absences': absences,
        'subject': subject
    }

    # building and preparing input dataframe
    df_input    = build_input_df(raw_input)
    risk_prob   = model.predict_proba(df_input)[0][1]
    prediction  = model.predict(df_input)[0]

    # --------------------------------------------------------
    # results layout
    # --------------------------------------------------------
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("🎯 Risk Assessment")
        if prediction == 1:
            st.error("⚠️ AT RISK")
            st.markdown("This student is predicted to be **at risk of academic failure.**")
        else:
            st.success("✅ PASSING")
            st.markdown("This student is predicted to be on track to **pass.**")

        # risk probability gauge
        st.metric(
            label = "Risk Probability",
            value = f"{risk_prob * 100:.1f}%",
            delta = f"{'High Risk' if risk_prob > 0.5 else 'Low Risk'}"
        )

        # risk level bar
        st.progress(float(risk_prob))
        st.caption(f"0% (No Risk) → 100% (Certain Risk)")

    with col2:
        st.subheader("📊 Risk Probability Gauge")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors  = ['#2ecc71' if risk_prob < 0.4
                   else '#f39c12' if risk_prob < 0.65
                   else '#e74c3c']
        ax.barh(['Risk'], [risk_prob], color=colors,
                height=0.4, edgecolor='white')
        ax.barh(['Risk'], [1 - risk_prob],
                left=[risk_prob], color='#ecf0f1',
                height=0.4, edgecolor='white')
        ax.axvline(x=0.5, color='black',
                   linestyle='--', linewidth=1.2,
                   label='Decision Threshold (0.5)')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Predicted Risk Probability', fontweight='bold')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("📋 Intervention Recommendation")
        if risk_prob >= 0.65:
            st.error("🔴 HIGH PRIORITY")
            st.markdown("""
            **Recommended Actions:**
            - Schedule immediate counsellor meeting
            - Contact parents or guardians
            - Assign academic support tutor
            - Review attendance records
            """)
        elif risk_prob >= 0.40:
            st.warning("🟡 MONITOR CLOSELY")
            st.markdown("""
            **Recommended Actions:**
            - Flag for monthly check-in
            - Encourage study group participation
            - Share available tutoring resources
            """)
        else:
            st.success("🟢 LOW RISK")
            st.markdown("""
            **Recommended Actions:**
            - No immediate intervention needed
            - Continue regular academic monitoring
            - Encourage extracurricular engagement
            """)

    st.divider()

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------
    st.subheader("🔍 Why Did the Model Make This Prediction?")
    st.markdown("""
    The chart below shows which features pushed this prediction toward 
    **At-Risk (red)** or **Passing (green)** and by how much.
    """)

    shap_values_input = explainer.shap_values(df_input)
    if isinstance(shap_values_input, list):
        shap_values_input = shap_values_input[1]

    # top 10 features by absolute SHAP value
    shap_series = pd.Series(
        shap_values_input[0],
        index=feature_names
    ).sort_values(key=abs, ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_colors = ['#e74c3c' if v > 0 else '#2ecc71'
                  for v in shap_series.values]
    ax2.barh(shap_series.index[::-1],
             shap_series.values[::-1],
             color=bar_colors[::-1],
             edgecolor='white')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_title('Top 10 Feature Contributions to This Prediction',
                  fontweight='bold')
    ax2.set_xlabel('SHAP Value (Red = Increases Risk, Green = Reduces Risk)')
    st.pyplot(fig2)
    plt.close()

    st.divider()

    # --------------------------------------------------------
    # student profile summary
    # --------------------------------------------------------
    st.subheader("📄 Student Profile Summary")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Academic Profile**")
        st.dataframe(pd.DataFrame({
            'Feature': ['Failures', 'Study Time',
                        'Absences', 'School Support'],
            'Value':   [failures, studytime, absences, schoolsup]
        }), hide_index=True, use_container_width=True)

    with col5:
        st.markdown("**Socioeconomic Profile**")
        st.dataframe(pd.DataFrame({
            'Feature': ["Mother's Education", "Father's Education",
                        'Family Support', 'Internet Access'],
            'Value':   [medu, fedu, famsup, internet]
        }), hide_index=True, use_container_width=True)

# ============================================================
# footer
# ============================================================
st.divider()
st.caption("""
Built as part of a Data Science portfolio project | 
Model: LightGBM with Optuna tuning | 
Explainability: SHAP TreeExplainer
""")