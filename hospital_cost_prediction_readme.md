
# Predicting Hospital Cost and Emergency Admissions Using Machine Learning

**Author:** Allison Evanich  
**Project Type:** Machine Learning / Healthcare Analytics  
**Tools:** Python, pandas, scikit-learn, matplotlib

---

## Project Overview

This project explores two healthcare machine learning tasks using a synthetic hospital admissions dataset:

1. Predicting **high-cost hospital admissions**
2. Predicting **emergency vs. non-emergency admissions**

A key goal of the analysis was not only to build predictive models, but also to evaluate model validity, detect **target leakage**, and interpret model limitations responsibly.

This project demonstrates an end-to-end machine learning workflow including:

- Data cleaning
- Feature engineering
- Model development
- ROC-based model evaluation
- Feature importance analysis
- Responsible interpretation of model performance

---

## Business Problem

Healthcare organizations often want to predict costly admissions and emergency utilization to improve:

- resource planning
- hospital staffing
- operational efficiency
- cost forecasting

This project evaluates whether demographic, administrative, and limited clinical variables can support those prediction tasks.

An important lesson from this project is that **strong model performance metrics are meaningless if the model contains target leakage**.

---

## Dataset

The dataset used in this project is a **synthetic healthcare dataset generated using Python's Faker library** and made available through Kaggle.

It includes information typically found in hospital admission records such as:

- Age
- Gender
- Blood Type
- Medical Condition
- Hospital
- Insurance Provider
- Admission Type
- Billing Amount
- Date of Admission
- Discharge Date
- Medication
- Test Results

Because the dataset is synthetic, it allows healthcare analytics experimentation without involving protected health information.

---

## Feature Engineering

Several additional variables were created to support the analysis.

### Length of Stay (LOS)

Length of stay was calculated using admission and discharge dates:

LOS = Discharge Date – Admission Date

### HighBilling Target

A binary target variable was created identifying admissions within the **top 25% of billing amounts**.

### EmergencyAdmission Target

A second binary variable was created identifying whether an admission was classified as **Emergency vs Non-Emergency**.

---

## Modeling Approach

Three machine learning models were developed during this project.

### 1. Leaky HighBilling Model

The first model attempted to predict whether an admission belonged to the top 25% of billing amounts.

However, the model mistakenly included **Billing Amount** as a predictor while it also defined the target variable. This produced near-perfect performance due to **target leakage**.

**Result:** ROC-AUC ≈ 1.0

This result is misleading because the model effectively had access to the answer.

---

### 2. Corrected HighBilling Model

To correct for leakage, the following variables were removed:

- Billing Amount
- Length of Stay
- other post-admission variables

After removing these features, the model was retrained using a Random Forest classifier.

**Result:** ROC-AUC ≈ 0.50

This indicates that the remaining demographic and administrative features contain little predictive signal for hospital cost.

---

### 3. Emergency Admission Model

A second modeling task attempted to predict whether an admission was classified as **Emergency vs Non-Emergency**.

To avoid leakage, the `Admission Type` column was removed from predictors.

**Model Results:**

- Accuracy ≈ 0.66
- ROC-AUC ≈ 0.50

The confusion matrix showed the model predicted **non-emergency admissions well but struggled to detect emergency cases**.

---

## Key Findings

Several important insights emerged from the project.

### Target Leakage Detection

The initial model's near-perfect performance revealed that the dataset contained features directly tied to the target variable.

Detecting and correcting this leakage was a critical step in producing a valid model.

### Limited Predictive Signal

After leakage removal, model performance dropped to near-random levels, indicating that the dataset lacked sufficient predictors for the modeling task.

In real healthcare settings, more predictive features might include:

- diagnosis codes
- procedure codes
- laboratory results
- imaging studies
- patient acuity indicators

### Importance of Responsible Model Evaluation

This project demonstrates that machine learning evaluation must go beyond simple accuracy metrics and include careful examination of feature relationships and data provenance.

---

## Ethical Considerations

Healthcare machine learning models must be developed responsibly.

Key ethical considerations highlighted by this project include:

- avoiding target leakage
- ensuring models are interpretable
- validating models before real-world deployment
- recognizing limitations of available data

Poorly validated models can lead to incorrect predictions that may influence healthcare decision-making.

---

## Repository Structure

hospital-cost-prediction-ml/
│
├── healthcare_project_cleaned.ipynb
├── README.md
│
├── figures/
│   ├── billing_distribution.png
│   ├── roc_leaky_highbilling.png
│   ├── feature_importance_leaky.png
│   ├── roc_corrected_highbilling.png
│   └── confusion_matrix_emergency.png
│
└── data/
    └── healthcare_dataset.csv

---

## Skills Demonstrated

- Data Cleaning
- Feature Engineering
- Machine Learning Modeling
- Random Forest Classification
- Model Evaluation (ROC-AUC)
- Feature Importance Interpretation
- Healthcare Analytics

---

## Future Improvements

Future iterations of this project could include:

- incorporating richer clinical variables
- comparing additional machine learning algorithms
- hyperparameter tuning
- feature selection techniques
- external validation datasets

These improvements could potentially improve predictive performance and model generalizability.

---

## Author

Allison Evanich  
Master of Science in Data Science
