# 22. Predict Car Prices using Random Forest



# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Sample dataset (replace with a real dataset if available)
data = {
  'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
  'model': ['Corolla', 'Civic', 'F-150', '3 Series', 'A4', 'Camry', 'Accord', 'Mustang', '5 Series', 'A6'],
  'year': [2015, 2017, 2018, 2016, 2015, 2018, 2016, 2015, 2017, 2018],
  'mileage': [50000, 30000, 40000, 60000, 45000, 35000, 55000, 65000, 30000, 20000],
  'price': [15000, 17000, 25000, 27000, 30000, 16000, 18000, 26000, 28000, 31000]
}

# Convert to dataframe
df = pd.DataFrame(data)

print(f'\nDataset: {df}')

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['make', 'model'])
print(f'\nEncoded Dataset: \n {df_encoded}')
print(f'\nColumns of Encoded Dataset: \n {df_encoded.columns}')
print('\nBasic Information of Encoded Dataset:\n')
print(f'{df_encoded.info()}')


# Define features (X) and target (y)
X = df_encoded.drop('price', axis=1)   # Features
y = df_encoded['price']   # target variable


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f'\nMean Squared Error: {mse}')
print(f'R2-Score: {r2}')