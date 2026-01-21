# 15. Credit card fraud detection using scikit-learn



# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the dataset (replace with your dataset path)
df = pd.read_csv('dataset/15_credit_card_fraud_detection/creditcard.csv')


# Display the first five rows of the dataset
print(f'Dataset Sample:\n {df.head()}')


# Separate features (X) and target (y)
X = df.drop('Class', axis=1)   # 'class' column is the target, with 0 for non-fraudalent transaction
y = df['Class']


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scale the features for numerical stability
scalar = StandardScaler()

# Fit the scalar on X_train and transforms it
X_train = scalar.fit_transform(X_train)

X_test = scalar.fit_transform(X_test)


# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'\nAccuracy: {accuracy * 100:.2f}%')
print(f'\nClassification Report: \n {report}')
print(f'\nConfusion Matrix: \n {conf_matrix}')