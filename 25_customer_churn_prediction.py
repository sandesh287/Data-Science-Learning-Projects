# 25. Customer Churn Prediction using classification algorithm (Logistic Regression)



# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv('dataset/25_customer_churn/customer_churn.csv')

# Display the first few rows of the dataset and dataset information
print(f'\nDataset Sample:\n {df.head()}')
print('\nBasic Dataset Information:\n')
print(df.info())


# Basic data preprocessing: handle missing values (if any)
df = df.dropna()


# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['gender', 'contract_type', 'payment_method'])

print(f'\nEncoded dataset sample:\n {df_encoded.head()}')


# Define features (X) and target (y)
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Feature scaling for numecial stability
scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Initialize te logistic regression classifier
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

# Display result
print(f'\nAccuracy: {accuracy * 100:.2f}%')
print(f'\nClassification Report:\n {report}')