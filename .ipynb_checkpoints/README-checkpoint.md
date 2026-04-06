# Student Academic Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> An end-to-end machine learning system that identifies secondary school 
> students at risk of academic failure — and explains exactly why.

---

## Project Overview

Most schools intervene **after** a student has already failed. This project 
builds an **early warning system** that flags at-risk students at the 
**start of the academic year** — before any exam results exist — using only 
demographic, socioeconomic, and behavioural data.

The system is:
- **Accurate**: tuned LightGBM with Bayesian optimisation
- **Explainable**: SHAP values for every individual prediction
- **Fair**: includes a gender fairness audit
- **Deployable**: live Streamlit web application

---

## Problem Statement

> Can student academic risk be predicted reliably from demographic, 
> socioeconomic, and behavioural features alone — without access to 
> current academic performance data?

**Target variable:** Binary classification
- `1` = At-Risk (final grade G3 < 10)
- `0` = Passing (final grade G3 ≥ 10)

---

## Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Citation | Cortez & Silva, 2008 |
| Records | 1,044 students |
| Features | 33 original + 9 engineered |
| Subjects | Mathematics + Portuguese Language |
| Target | Binary (G3 < 10 = at-risk) |

[Download Dataset](https://archive.uci.edu/dataset/320/student+performance)

---

## Project Structure
student_risk_project/
│
├── data/
│   ├── raw/                    ← original unmodified data
│   └── processed/              ← cleaned and engineered datasets
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_explainability.ipynb
│
├── models/
│   ├── lgbm_final.pkl          ← trained LightGBM model
│   ├── scaler.pkl              ← fitted StandardScaler
│   └── feature_names.pkl      ← ordered feature list for deployment
│
├── reports/
│   └── research_report.md     ← graduate-style research report
│
├── visuals/                    ← all saved charts and figures
│
├── app/
│   └── app.py                 ← Streamlit web application
│
├── README.md
└── requirements.txt


---

## Methodology

### Pipeline Overview

    Raw Data → Cleaning → EDA → Feature Engineering →
    Modelling → Evaluation → Explainability → Deployment

### Key Design Decisions

**1. Leakage Prevention**
First and second period grades (G1, G2) were excluded entirely. 
These grades are unavailable at prediction time — including them 
would produce an artificially high accuracy but a system that 
cannot be deployed.

**2. Class Imbalance Handling**
SMOTE was applied exclusively to the training set to avoid 
inflating evaluation metrics on the test set.

**3. Feature Engineering**
Nine domain-informed features were created from the original 
variables — including an interaction term between past failures 
and absences that became the strongest single predictor.

**4. Model Selection**
Five models were compared on F1-score. LightGBM was selected 
and tuned using Optuna Bayesian optimisation over 100 trials 
with 5-fold stratified cross validation.

**5. Explainability**
SHAP TreeExplainer provides both global feature importance and 
individual-level prediction explanations — essential for 
real-world deployment in an institutional setting.

---

## Results

| Metric | Score |
|---|---|
| F1 Score | *0.3855* |
| Recall | *0.3478* |
| Precision | *0.4324* |
| ROC-AUC | *0.7679* |

> **Primary metric: F1-Score and Recall.**
> Recall is prioritised — missing a genuinely at-risk student 
> is more costly than a false alarm.

---

## Feature Engineering

| Engineered Feature | Source Columns | Reasoning |
|---|---|---|
| `avg_alcohol` | Dalc, Walc | Overall alcohol consumption pattern |
| `parent_edu_avg` | Medu, Fedu | Combined parental education signal |
| `parent_edu_max` | Medu, Fedu | Most educated parent's influence |
| `total_support` | famsup, schoolsup, paid | Available support systems count |
| `social_score` | famrel, freetime, goout | Social engagement composite |
| `health_risk_score` | health, avg_alcohol, absences | Lifestyle risk composite |
| `is_first_gen` | Medu, Fedu | First generation student flag |
| `failure_absence_interaction` | failures, absences | Compounding risk interaction term |
| `study_support_ratio` | studytime, total_support | Effort relative to support |

---

## Models Compared

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable baseline |
| Random Forest | Ensemble baseline |
| XGBoost | Gradient boosting comparison |
| **LightGBM** | **Final selected model** |

---

## Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to explain 
every prediction at two levels:

**Global**: Which features drive risk across all students

**Local**: Why the model flagged this specific student

This means a school counsellor receives not just a risk score — 
but a ranked list of the exact factors behind it.

---

## Fairness Audit

A basic fairness audit compares actual vs predicted risk rates 
across gender groups to check whether the model introduces or 
amplifies bias.

Predicted risk rates closely mirror actual risk rates for both 
male and female students — confirming the model does not add 
disparity beyond what exists in the original data.

---

## Streamlit Application

The deployed application allows school counsellors to:

1. Enter a complete student profile via sidebar inputs
2. Receive an instant risk prediction with probability score
3. View a colour-coded intervention recommendation
4. See the top 10 SHAP features driving the prediction
5. Review a clean summary of the student profile

**To run the app locally:**
```bash
# clone the repository
git clone https://github.com/yourusername/student-risk-project.git
cd student-risk-project/app

# install dependencies
pip install -r requirements.txt

# launch the app
python -m streamlit run app.py
```

---

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, LightGBM, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Deployment | Streamlit |
| Environment | Conda |

---

## Notebooks Guide

| Notebook | Description |
|---|---|
| `01_data_acquisition` | Load, inspect, and validate raw data |
| `02_data_cleaning` | Encode, scale, and remove leakage |
| `03_eda` | Visual and statistical exploration |
| `04_feature_engineering` | Create and validate 9 new features |
| `05_modelling` | Train, tune, and evaluate all models |
| `06_explainability` | SHAP analysis and fairness audit |

---

## Research Report

A graduate-style research report covering problem motivation, 
methodology, results, limitations, and future work is available at:

[`reports/research_report.md`](reports/research_report.md)

---

## Limitations

- Dataset is limited to two Portuguese secondary schools
- 1,044 records is small by modern ML standards
- Fairness audit covers gender only — broader audit recommended
- Model identifies correlates of risk, not causal factors
- Generalisation to other educational systems requires validation

---

## Future Work

- Longitudinal validation across multiple academic years
- Formal fairness constraints during model training
- Causal inference modelling
- Integration with learning management system data
- Decision threshold optimisation based on institutional cost ratios

---

## Author

**Paul Ahossey**
- GitHub: https://github.com/Ahossey
- LinkedIn: https://www.linkedin.com/in/paulahossey/
- Email: paulahossey@gmail.com

---

## References

Cortez, P., & Silva, A. (2008). Using data mining to predict 
secondary school student performance. FUBUTEC 2008.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to 
interpreting model predictions. NeurIPS 2017.

Chawla, N. V., et al. (2002). SMOTE: Synthetic minority 
over-sampling technique. JAIR, 16, 321-357.

UCI ML Repository: Student Performance Dataset.
https://archive.uci.edu/dataset/320/student+performance

---

*This project was built as a portfolio demonstration of end-to-end 
data science competence — from raw data to deployed application.*
