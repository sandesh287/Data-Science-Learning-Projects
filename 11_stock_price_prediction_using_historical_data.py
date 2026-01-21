# 11. Stock price prediction using historical data



# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Sample historical stock price data
data = {
  'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
  'Close': [150, 152, 153, 155, 154, 156, 157, 158, 159, 160]
}


# Converting data into dataframe
df = pd.DataFrame(data)


# Convert date column to numerical format or ordinal, for regression analysis
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

print('Stock Price Data:')
print(df.head())


# Set features and target variables
X = df[['Date']]
y = df['Close']


# Split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the linear regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display
print('\n')
print(f'Mean Squared Error: {mse}')
print(f'R2-Score: {r2}')

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, label='Actual Stock Prices', marker='o')
plt.plot(X_test, y_pred, label='Predicted Stock Prices', marker='x')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()


# Test model with new future date
future_date = pd.Timestamp('2024-01-11').toordinal()

future_date = pd.DataFrame({'Date': [future_date]})

predicted_price = model.predict(future_date)
print(f'Predicted Stock Price: ${predicted_price[0]:.2f}')