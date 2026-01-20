# 5 Human activity recognition using smartphones dataset with Random Forest



# importing libraries
import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


# Download and load dataset
# https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

print('\n')
print("Path to dataset files:", path)

# Build full path to train.csv, test.csv
train_csv_path = os.path.join(path, 'train.csv')
test_csv_path = os.path.join(path, 'test.csv')

# Read train.csv, test.csv
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

print('\n')
print(f'Train Dataset shape: {df_train.shape}')
print(f'Test Dataset shape: {df_test.shape}')


# Preprocess the dataset
# Separate features and labels for train data
X_train = df_train.drop('Activity', axis=1)
y_train = df_train['Activity']

# Separate features and labels for test data
X_test = df_test.drop('Activity', axis=1)
y_test = df_test['Activity']


# Initialize and train the random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


# Make prediction on the test set
y_pred = model.predict(X_test)


# Evaluate the model using accuracy, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('\n')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')


# Visualize the confusion matrix using seaborns heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()