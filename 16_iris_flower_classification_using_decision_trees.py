# 16. iris Flower classification using Decision Trees



# import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Load Iris dataset
iris = load_iris()

# Create a dataframe from iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['species'] = iris.target   # add the species column (target)

# Display the first few rows of the dataset
print(f'Iris Dataset:\n {df.head()}')

# Split the dataset into features (X) and target (y)
X = df.drop('species', axis=1)   # features (sepal length, sepal width, petal length, petal width)
y = df['species']   # Target (species: 0 = Setosa, 1 = Versicolor, 2 = Virginica)


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize and train a decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred)

# Display results
print(f'\nAccuracy: {accuracy * 100:.2f}%')
print(f'\nConfusion Matrix: \n {conf_matrix}')
print(f'\nClassification Report: \n {class_report}')

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree for Iris Flower Classification')
plt.show()