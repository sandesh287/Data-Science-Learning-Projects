# 28. Disease prediction using ML algorithms
# Dataset url: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset



# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# Ignore warnings
warnings.filterwarnings('ignore')


# 1. Load the dataset
df = pd.read_csv('dataset/28_disease_prediction/heart_disease.csv')

# Display first few rows
print(f'\nDataset Sample:\n {df.head()}')

# Display basic information
print('\nBasic Dataset Information:\n')
print(df.info())


# 2. Data Preprocessing

# Check if there is any missing values
print(f'\nMissing Values:\n {df.isnull().sum()}')


# Feature Scaling
scalar = StandardScaler()
scaled_features = scalar.fit_transform(df.drop('target', axis=1))

X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['target']


# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 4. Train Multiple ML Models

# 1. Logistic Regression
log_model = LogisticRegression()

log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_preds)

print(f'\nLogistic Regression Accuracy: {log_accuracy * 100:.2f}%')


# 2. Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)

print(f'\nRandom Forest Accuracy: {rf_accuracy * 100:.2f}%')


# 5. Evaluate the Best Model
best_model = rf_model if rf_accuracy > log_accuracy else log_model
best_preds = rf_preds if rf_accuracy > log_accuracy else log_preds

print('\nBest Model Metrics:\n')
print('Accuracy Score: ', accuracy_score(y_test, best_preds))
print('\nClassification Report:\n', classification_report(y_test, best_preds))
print('\nConfusion Matrix: \n', confusion_matrix(y_test, best_preds))


# 6. Visualize the Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# 7. Make predictions on new data
new_data = pd.DataFrame({
  'age': [45],
  'sex': [1],
  'cp': [2],
  'trestbps': [130],
  'chol': [230],
  'fbs': [0],
  'restecg': [1],
  'thalach': [150],
  'exang': [0],
  'oldpeak': [0.5],
  'slope': [2],
  'ca': [0],
  'thal': [2],
})

# Scale new data
new_data_scaled = scalar.transform(new_data)

prediction = best_model.predict(new_data_scaled)

print('\nPrediction for New Data:', 'At Risk of Heart Disease' if prediction[0] == 1 else 'No Heart Disease')