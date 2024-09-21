# Coronary Heart Disease (CHD) Risk Prediction

This project predicts the 10-year risk of coronary heart disease (CHD) based on demographic, behavioral, and medical risk factors. The dataset includes information such as age, sex, smoking habits, cholesterol levels, blood pressure, BMI, and more. Using this dataset, we developed machine learning models to identify individuals at high risk of developing CHD.

## Project Overview

### Dataset Description

The dataset contains the following variables:

- **Demographic:**
  - **Sex**: Male or female (Nominal)
  - **Age**: Age of the patient (Continuous)

- **Behavioral:**
  - **Current Smoker**: Whether the patient is a current smoker (Nominal)
  - **Cigs Per Day**: Number of cigarettes smoked per day (Continuous)

- **Medical History:**
  - **BP Meds**: Whether the patient was on blood pressure medication (Nominal)
  - **Prevalent Stroke**: Whether the patient had previously had a stroke (Nominal)
  - **Prevalent Hyp**: Whether the patient was hypertensive (Nominal)
  - **Diabetes**: Whether the patient had diabetes (Nominal)

- **Medical (Current):**
  - **Tot Chol**: Total cholesterol level (Continuous)
  - **Sys BP**: Systolic blood pressure (Continuous)
  - **Dia BP**: Diastolic blood pressure (Continuous)
  - **BMI**: Body Mass Index (Continuous)
  - **Heart Rate**: Heart rate (Continuous)
  - **Glucose**: Glucose level (Continuous)

- **Predictive Target:**
  - **10-year risk of CHD**: Binary classification – “1” (Yes) for high risk, “0” (No) for low risk

---

## Workflow

### 1. **Data Exploration**
   - Conducted an initial exploration of the dataset to understand variable distributions, detect missing values, and compute basic statistical measures (mean, median, standard deviation, etc.).
   - Visualized continuous and categorical data to identify patterns and relationships.

### 2. **Algorithm Identification**
   - The project focuses on solving a binary classification problem. The following machine learning algorithms were identified for comparison:
     - **Logistic Regression**: A simple yet effective model for binary classification.
     - **Random Forest**: A versatile, non-linear ensemble method that handles complex relationships and imbalanced data.
     - **Support Vector Machine (SVM)**: A powerful algorithm for distinguishing between two classes, even when the data is not linearly separable.

### 3. **Data Preprocessing**
   - **Missing Values**: Handled using imputation techniques such as mean/mode filling.
   - **Encoding**: Converted categorical variables like gender and smoking status into numerical format using one-hot encoding.
   - **Scaling**: Normalized continuous variables (e.g., Age, Cholesterol, BMI) to ensure uniform input to the models.
   - **Data Splitting**: Split the data into training and testing sets to evaluate model performance.

### 4. **Model Development**
   - Created three different machine learning models:
     - **Logistic Regression**
     - **Random Forest**
     - **Support Vector Machine (SVM)**

### 5. **Model Evaluation**
   - Evaluated each model using metrics such as **Precision**, **Recall**, and **F1-Score**.
   - Analyzed the confusion matrix to assess the trade-off between true positives and false negatives.

### 6. **Model Comparison & Selection**
   - Performed a comparative analysis of all models based on precision, recall, and F1-score.
   - **Random Forest** was selected as the best model due to its superior handling of imbalanced data and higher F1-score. This model effectively captured the complexity of interactions between risk factors.

### 7. **Saving the Model**
   - Saved the trained Random Forest model using `joblib` or `pickle` for future use in the Streamlit application.

### 8. **Streamlit Web App**
   - Developed a simple web application using **Streamlit**, allowing users to input personal data such as age, cholesterol, and smoking status.
   - The app predicts whether the user is at risk of CHD in the next 10 years.

### 9. **Deployment & GitHub**
   - Uploaded the project files, including the Jupyter notebook, trained model, and Streamlit app, to GitHub.
   - Provided instructions for users to run the app locally or deploy it on their own platforms.

---

## How to Run the Project

### Requirements
To run this project, you'll need the following Python packages:

```bash
pip install pandas numpy scikit-learn streamlit
```

### Instructions
1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/username/CHD_Risk_Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CHD_Risk_Prediction
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Files:
- **CHD_Prediction.ipynb**: The Jupyter notebook containing the entire workflow, from data exploration to model evaluation.
- **model.pkl**: The saved Random Forest model used for prediction.
- **app.py**: The Streamlit web application code.
  
---

## Model Performance

| Metric          | Logistic Regression | Random Forest | SVM         |
|-----------------|---------------------|---------------|-------------|
| **Precision**   | X.XX                | X.XX          | X.XX        |
| **Recall**      | X.XX                | X.XX          | X.XX        |
| **F1-Score**    | X.XX                | X.XX          | X.XX        |

The Random Forest model was chosen for its balance between precision and recall, resulting in the best F1-score, making it the most effective at predicting the 10-year risk of CHD.

---

## Conclusion

This project successfully predicts the 10-year risk of coronary heart disease using machine learning models. By comparing multiple models, we identified Random Forest as the best performing algorithm. The web app allows users to input their data and receive predictions, making this tool both accessible and practical.

---
