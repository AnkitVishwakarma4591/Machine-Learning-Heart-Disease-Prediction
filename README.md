# Coronary Heart Disease (CHD) Risk Prediction

> Predict the 10-year risk of developing coronary heart disease based on clinical, behavioral, and demographic indicators.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://machine-learning-heart-disease-prediction-8.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Live Application

**[https://machine-learning-heart-disease-prediction-8.streamlit.app/](https://machine-learning-heart-disease-prediction-8.streamlit.app/)**

---

## Project Overview

This project predicts the 10-year risk of coronary heart disease (CHD) based on demographic, behavioral, and medical risk factors. The dataset includes information such as age, sex, smoking habits, cholesterol levels, blood pressure, BMI, and more. Six machine learning algorithms were trained and evaluated, with **XGBoost** selected as the final deployed model.

---

## Dataset Description

The dataset contains **4,238 patient records** with the following variables:

### Demographic
| Variable | Type | Description |
|----------|------|-------------|
| Sex | Nominal | Male or Female |
| Age | Continuous | Age of the patient |

### Behavioral
| Variable | Type | Description |
|----------|------|-------------|
| Current Smoker | Nominal | Whether the patient currently smokes |
| Cigs Per Day | Continuous | Number of cigarettes smoked per day |

### Medical History
| Variable | Type | Description |
|----------|------|-------------|
| BP Meds | Nominal | Whether on blood pressure medication |
| Prevalent Stroke | Nominal | Whether the patient had a prior stroke |
| Prevalent Hyp | Nominal | Whether the patient is hypertensive |
| Diabetes | Nominal | Whether the patient has diabetes |

### Medical (Current)
| Variable | Type | Description |
|----------|------|-------------|
| Tot Chol | Continuous | Total cholesterol level |
| Sys BP | Continuous | Systolic blood pressure |
| Dia BP | Continuous | Diastolic blood pressure |
| BMI | Continuous | Body Mass Index |
| Heart Rate | Continuous | Heart rate in BPM |
| Glucose | Continuous | Glucose level |

### Target Variable
| Variable | Description |
|----------|-------------|
| TenYearCHD | 10-year CHD risk — `1` (High Risk) / `0` (Low Risk) |

---

## Workflow

### 1. Data Exploration
- Explored variable distributions, detected missing values, and computed statistical measures.
- Visualized continuous and categorical data to identify patterns and relationships.

### 2. Data Preprocessing
- **Missing Values**: Imputed using column mean values.
- **Encoding**: Categorical variables (sex, smoking status) converted to numerical format.
- **Normalization**: MinMaxScaler applied to: `age`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`, `cigsPerDay`.
- **Data Splitting**: 97% training / 3% testing split.

### 3. Model Development
Six algorithms were trained and compared:
- Logistic Regression (LOR)
- K-Nearest Neighbours (KNN)
- Decision Tree (DT)
- Random Forest (RF)
- Gradient Boosting (GB)
- **XGBoost (XGB)** ← Selected Model

### 4. Model Evaluation
Each model was evaluated using Accuracy, Precision, Recall, and F1-Score.

### 5. Model Saving
The final XGBoost model was saved using `joblib` for deployment in the Streamlit application.

---

## Model Performance

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 85.37% |
| K-Nearest Neighbours | 86.18% |
| Decision Tree | 82.93% |
| Random Forest | 86.18% |
| Gradient Boosting | 86.18% |
| **XGBoost** | **84.55%** |

### Precision, Recall & F1-Score

| Metric | LOR | KNN | DT | RF | GB | **XGB** |
|--------|-----|-----|----|----|-----|---------|
| **Precision** | 0.00% | 100.00% | 35.71% | 50.00% | 50.00% | **33.33%** |
| **Recall** | 0.00% | 0.00% | 29.41% | 5.88% | 5.88% | **5.88%** |
| **F1-Score** | 0.00% | 0.00% | 32.26% | 10.53% | 10.53% | **17.39%** |

> XGBoost was selected as the final model for its best F1-Score among all candidates, making it most effective at balancing precision and recall for CHD risk prediction.

---

## How to Run the Project

### Requirements

```bash
pip install pandas numpy scikit-learn streamlit xgboost joblib
```

### Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/AnkitVishwakarma4591/Machine-Learning-Heart-Disease-Prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Machine-Learning-Heart-Disease-Prediction
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Repository Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `Heart Disease Model.ipynb` | Full notebook: EDA, preprocessing, training, evaluation |
| `heart_disease_model.pkl` | Saved XGBoost model |
| `heart_disease_data.csv` | Dataset (4,238 records) |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime specification |

---

## Conclusion

This project successfully predicts the 10-year risk of coronary heart disease using six machine learning models. After comparative evaluation, XGBoost was selected as the final deployed model for its best F1-score. The Streamlit web app allows users to input their clinical and lifestyle data and receive an instant CHD risk prediction.

---

## Contact

**Developed by:** Ankit Vishwakarma

| | |
|--|--|
| Email | [ankitvishwakarma4591@gmail.com](mailto:ankitvishwakarma4591@gmail.com) |
| Phone | +91 9060782203 |
| LinkedIn | [Ankit Vishwakarma](https://www.linkedin.com/in/ankit-vishwakarma-324baa2a6/) |