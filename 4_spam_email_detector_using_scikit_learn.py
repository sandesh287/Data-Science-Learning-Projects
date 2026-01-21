# 4. Spam email detector using scikit-learn



# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset and split it into training and testing sets
data = pd.read_csv('dataset/4_spam_not_spam_email/spam.csv')
X = data.drop('spam', axis=1)
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the Logistic Regression model to classify emails as spam or not spam
model = LogisticRegression(max_iter=10000)

# OR
# model = Pipeline([
#   ('scalar', StandardScaler()),
#   ('logreg', LogisticRegression(max_iter=10000))
# ])

model.fit(X_train, y_train)


# Predict the labels spam or not spam on test set
y_pred = model.predict(X_test)

print('\n')
print(X_test)


# Evaluate the model using accuracy, confusion matrix, precision, recall and F1 score
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print('\n')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')


# Calculate and Visualize the confusion matrix using Seaborn's heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()