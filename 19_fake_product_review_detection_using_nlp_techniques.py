# 19. Fake Product review detection using NLP techniques
# The goal is to identify whether a given product review is genuine or fake based on its text content.
# Dataset url: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus


# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset (replace with the actual dataset path)
df = pd.read_csv('dataset/19_fake_product_review_detection/deceptive-opinion.csv')

# Display the first 5 rows of the dataset
print(f'\nDataset Sample:\n {df.head()}')


# Define features (X) and target (y)
X = df['text']   # assuming the dataset has a 'text' column
y = df['deceptive']   # assuming the dataset has a 'deceptive' column with values 'deceptive' or 'truthful'


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Initialize the TfidfVectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform
X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# Initialize the logistic regression classifier
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'\nAccuracy: {accuracy * 100:.2f}%')
print(f'\nClassification Report:\n {report}')