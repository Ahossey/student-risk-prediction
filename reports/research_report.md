# Predicting Student Academic Risk Using Machine Learning
### A Data Science Project
**Author:** AHOSSEY PAUL
**Date:** 2025
**Dataset:** UCI Student Performance Dataset
**Repository:** github.com/Ahossey/student-risk-project

---

## Abstract

This project develops a machine learning system to identify secondary 
school students at risk of academic failure using **demographic, 
socioeconomic**, and **behavioural** data collected at the start of the 
academic year. A LightGBM classifier tuned with Bayesian optimisation 
achieves strong predictive performance on held-out test data, 
outperforming baseline models including Logistic Regression, 
Decision Tree, and Random Forest. SHAP values provide both global 
and individual-level explanations of model predictions, making the 
system interpretable and deployable in a real institutional context. 
A basic fairness audit confirms the model does not introduce 
systematic bias across gender groups. The project concludes with 
a Streamlit web application that allows school counsellors to 
generate instant risk assessments for individual students.

---

## 1. Introduction and Motivation

Academic failure is one of the most consequential and preventable 
outcomes in secondary education. Students who fail their final 
examinations face limited access to higher education, reduced 
lifetime earnings, and diminished social mobility. Despite the 
availability of data on student behaviour and background, most 
schools rely on reactive rather than proactive interventions — 
acting only after failure has already occurred.

This project addresses that gap by building an early warning system 
that identifies at-risk students before final grades are determined. 
By relying exclusively on features available at the start of the 
academic year — demographic background, family circumstances, 
study habits, and past academic history — the system can flag 
students early enough for meaningful intervention.

The central research question is:

> Can student academic risk be predicted reliably from demographic, 
> socioeconomic, and behavioural features alone, without access to 
> current academic performance data?

---

## 2. Dataset

The UCI Student Performance Dataset was collected from two Portuguese 
secondary schools and covers students in Mathematics and Portuguese 
Language courses. The combined dataset contains 1,044 student records 
and 33 features spanning four categories.

| Category | Features |
|---|---|
| Demographics | Age, sex, urban/rural address |
| Socioeconomic | Parental education, parental occupation |
| Behavioural | Study time, absences, alcohol consumption |
| Academic History | Past failures, extra support, tutoring |

The target variable — academic risk — was derived from the final 
grade (G3) using a threshold of G3 < 10, consistent with the 
Portuguese grading system where 10 out of 20 represents the 
minimum passing score. This produced a binary classification 
problem with 32.9% at-risk and 67.1% passing students.

Importantly, the first and second period grades (G1 and G2) were 
excluded from all models. Including them would constitute data 
leakage — these grades are unavailable at the start of the 
semester when early intervention decisions must be made.

---

## 3. Methodology

### 3.1 Data Preprocessing

The raw dataset required minimal cleaning — no missing values or 
duplicate records were present. Categorical variables were encoded 
using label encoding for binary features and one-hot encoding for 
multi-class features. Numerical features were standardised using 
StandardScaler. All preprocessing was applied to training data only 
and then applied to validation and test sets to prevent data leakage.

### 3.2 Feature Engineering

Nine domain-informed features were engineered from the original 
variables to capture compound risk signals:

| Feature | Rationale |
|---|---|
| avg_alcohol | Average of weekday and weekend consumption |
| parent_edu_avg | Combined parental education signal |
| parent_edu_max | Influence of the more educated parent |
| total_support | Count of available support systems |
| social_score | Composite social engagement indicator |
| health_risk_score | Compound lifestyle risk measure |
| is_first_gen | Binary first-generation student flag |
| failure_absence_interaction | Interaction term capturing compounding risk |
| study_support_ratio | Study effort relative to support access |

The interaction term failure_absence_interaction produced the 
strongest correlation with the target variable among all engineered 
features, validating the hypothesis that compounding risk factors 
carry more predictive signal than individual features alone.

### 3.3 Class Imbalance

The 67/33 class split was addressed using SMOTE (Synthetic Minority 
Oversampling Technique) applied exclusively to the training set. 
SMOTE generates synthetic minority class samples by interpolating 
between existing at-risk student records. Applying SMOTE only to 
the training set ensures that evaluation metrics reflect true 
generalisation performance on the real-world class distribution.

### 3.4 Model Selection

Five classification models were trained and compared on the 
validation set using F1-score as the primary metric:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

LightGBM demonstrated the strongest balance of F1-score, Recall, 
and ROC-AUC and was selected for hyperparameter tuning.

### 3.5 Hyperparameter Tuning

LightGBM hyperparameters were tuned using Optuna with 100 trials 
of Bayesian optimisation and 5-fold stratified cross validation. 
Optuna was selected over GridSearchCV for its efficiency — it 
intelligently focuses search effort on promising regions of the 
hyperparameter space rather than exhaustively evaluating all 
combinations.

---

## 4. Results

### 4.1 Model Performance

The tuned LightGBM model achieved the following performance on 
the held-out test set:

| Metric | Score |
|---|---|
| F1 Score | see notebook output |
| Recall | see notebook output |
| Precision | see notebook output |
| ROC-AUC | see notebook output |

F1-score and Recall are the most meaningful metrics in this 
context. Recall measures the proportion of genuinely at-risk 
students successfully identified by the model — minimising 
false negatives is the primary objective of an early warning system.

### 4.2 Feature Importance

SHAP analysis identified the following features as most influential 
in driving model predictions globally:

- failure_absence_interaction — the engineered interaction term 
  between past failures and absences
- failures — number of past class failures
- absences — total school absences
- health_risk_score — composite lifestyle risk indicator
- studytime — weekly study hours

These findings are consistent with the patterns identified during 
exploratory data analysis and align with established research on 
determinants of academic performance.

### 4.3 Fairness Analysis

A basic fairness audit comparing actual and predicted risk rates 
across gender groups confirmed that the model does not introduce 
systematic bias beyond the distributions present in the original 
data. Predicted risk rates closely mirror actual risk rates for 
both male and female students.

---

## 5. Limitations

Several limitations should be acknowledged:

**Dataset size:** With 1,044 records the dataset is small by 
modern machine learning standards. Performance estimates may 
have higher variance than would be observed with larger samples.

**Geographic specificity:** The data was collected from two 
Portuguese secondary schools. Generalisation to other educational 
systems, countries, or cultural contexts cannot be assumed without 
validation on new data.

**Proxy discrimination:** Although the fairness audit found no 
systematic gender bias, features such as parental education and 
address type may act as proxies for socioeconomic status in ways 
that disadvantage already marginalised groups. A more comprehensive 
fairness analysis would examine multiple protected attributes and 
apply formal fairness metrics.

**Causal inference:** The model identifies correlates of academic 
risk — not causes. A student flagged due to high absences may be 
absent because of an underlying health condition, not because 
absences themselves cause failure. Interventions must be designed 
with this distinction in mind.

---

## 6. Future Work

Several extensions would strengthen this project:

- **Longitudinal validation** — testing the system on new cohorts 
  across multiple academic years to assess temporal stability
- **Formal fairness constraints** — applying algorithmic fairness 
  methods such as equalized odds or demographic parity during 
  model training
- **Causal modelling** — using causal inference frameworks to 
  distinguish correlates from genuine risk factors
- **Richer feature sets** — incorporating classroom engagement 
  data, assignment submission patterns, or learning management 
  system logs
- **Threshold optimisation** — calibrating the decision threshold 
  based on the actual cost ratio of false negatives to false 
  positives in a specific institutional context

---

## 7. Conclusion

This project demonstrates that student academic risk can be 
predicted with meaningful accuracy from demographic, socioeconomic, 
and behavioural features available at the start of the academic 
year. The combination of rigorous preprocessing, domain-informed 
feature engineering, ensemble modelling, and SHAP-based 
explainability produces a system that is both technically sound 
and practically deployable.

The inclusion of a fairness audit and an honest discussion of 
limitations reflects the ethical responsibilities that accompany 
predictive modelling in high-stakes domains. An early warning 
system is only beneficial if it is accurate, interpretable, 
fair, and used to support students — not to label or stigmatise them.

---

## References

Cortez, P., & Silva, A. (2008). Using data mining to predict 
secondary school student performance. In Proceedings of 5th 
Annual Future Business Technology Conference (pp. 5-12).

UCI Machine Learning Repository. Student Performance Dataset. 
https://archive.uci.edu/dataset/320/student+performance

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to 
interpreting model predictions. Advances in Neural Information 
Processing Systems, 30.

Chawla, N. V., et al. (2002). SMOTE: Synthetic minority 
over-sampling technique. Journal of Artificial Intelligence 
Research, 16, 321-357.