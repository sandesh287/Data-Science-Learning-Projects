# 8. Predict house prices with Linear Regression



# import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load the California housing dataset
housing = fetch_california_housing(as_frame=True)


# Create a Dataframe from the dataset
df = housing.frame

print('\n')
print('California Housing Data:')
print(df.head())


# Features (Independent variable) and target (dependent variable)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model using MSE and R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\n')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

print('\n')
print('Model Cofficients: ')
print(f'Intercept: {model.intercept_}')
print(f'Coefficients:\n {model.coef_}')

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print('\n')
print('Coefficients for each feature: ')
print(coef_df)


# Test the model with new data
new_data = pd.DataFrame({
  'MedInc': [5],
  'HouseAge': [30],
  'AveRooms': [6],
  'AveBedrms': [1],
  'Population': [500],
  'AveOccup': [3],
  'Latitude': [34.05],
  'Longitude': [-118.35]
})

predicted_price = model.predict(new_data)

print('\n')
print(f'Predicted House Price: ${predicted_price[0]:.2f}')