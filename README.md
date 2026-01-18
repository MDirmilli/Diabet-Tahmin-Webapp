# Diabetes Prediction with Machine Learning

This project demonstrates a binary classification pipeline for predicting diabetes using multiple machine learning algorithms and automatically selecting the best-performing model based on accuracy.

The dataset is synthetically generated and includes basic health indicators such as Age, BMI, Blood Pressure, and Glucose level.

---

## Project Overview

- Synthetic dataset generation
- Training multiple machine learning models
- Model comparison using accuracy metric
- Automatic best model selection
- Saving the best model using Joblib

This project is suitable for:
- Machine Learning course assignments
- Model comparison demonstrations
- Beginner–intermediate ML portfolios

---

## Machine Learning Models Used

The following algorithms are implemented and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes
- Gradient Boosting
- Extra Trees Classifier
- XGBoost
- Multi-Layer Perceptron (Neural Network)

---

## Dataset Description

The dataset file is named veri.csv and contains the following columns:

Age – Patient age  
BMI – Body Mass Index  
BloodPressure – Blood pressure value  
Glucose – Blood glucose level  
Diabetes – Target label (0 = No, 1 = Yes)

The dataset is generated using simple rule-based logic.  
High glucose levels and high BMI values increase the likelihood of diabetes.

---

## Project Workflow

1. Generate synthetic health data
2. Save dataset as CSV
3. Load dataset and split into training and test sets
4. Train multiple machine learning models
5. Evaluate models using accuracy
6. Select the best-performing model
7. Save the best model to disk

---

## Model Performance Example

Logistic Regression: 0.88 Accuracy  
SVM: 0.88 Accuracy  
K-Nearest Neighbors: 0.85 Accuracy  
Decision Tree: 0.99 Accuracy  
Gaussian Naive Bayes: 0.93 Accuracy  
Gradient Boosting: 0.99 Accuracy  
Extra Trees: 0.99 Accuracy  
XGBoost: 0.98 Accuracy  
MLP (Neural Network): 0.87 Accuracy  

Best Model:  
Decision Tree (99% Accuracy)

---

## Saved Model

The best-performing model is saved as:

eniyi.joblib

This model can be loaded later for prediction without retraining.

---

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

---

## How to Run

1. Install required dependencies:
pip install pandas numpy scikit-learn xgboost joblib

2. Run the dataset generation script
3. Run the model training and evaluation script
4. Observe model comparison results in the console
5. Use eniyi.joblib for inference

---

## Notes

- This dataset is synthetic and intended for educational purposes only
- High accuracy is expected due to rule-based data generation
- Real medical datasets should be used for real-world applications

---

## License

This project is intended for educational and academic use.
