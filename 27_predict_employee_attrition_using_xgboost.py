# 27. Predict Employee Attrition using XGBoost
# In this project, we will build a predictive model using XGBoost to determine whether an employee is likely to leave a company based on various job related features, such as job satisfaction, years at the company, salary and performance ratings.
# Dataset url: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset



# import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv('dataset/27_employee_attrition/Attrition.csv')

# Display the first few rows
print(f'\nDataset Preview:\n {df.head()}')

# Display dataset information
print('\nDataset Information:\n')
print(df.info())


# Preprocess the dataset

# Drop irrelevant columns
df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode the categorical variable
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
  df[column] = label_encoder.fit_transform(df[column])


# Split the features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Create and train the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)

# Make prediction
y_pred = model.predict(X_test)

# Evaluate the model on test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')
print(f'\nClassification Report:\n {class_report}')
print(f'\nConfusion Matrix:\n {conf_matrix}')

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()