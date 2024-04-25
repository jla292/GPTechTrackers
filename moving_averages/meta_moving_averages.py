# import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
warnings.filterwarnings('ignore')

# import the closing price data of META stock for the period of 2 years -

meta_df = pd.read_csv('data/META.csv')
meta_df.head()

# Create 20 days simple moving average column

# convert Date column to datetime
meta_df['Date'] = pd.to_datetime(meta_df['Date'], format = '%Y-%m-%d')
# sort by datetime
meta_df.sort_values(by='Date', inplace=True, ascending=True)

# Create 50 days simple moving average column
meta_df['50_SMA'] = meta_df['Close'].rolling(window = 50, min_periods = 1).mean()
meta_df.dropna()
meta_df.head()

# visualize data

# Calculate the number of quarters in the data range
num_quarters = (meta_df['Date'].dt.year.max() - meta_df['Date'].dt.year.min() + 1) * 4

meta_df['Adj Close'].plot(figsize = (15, 8), fontsize = 12)
plt.grid()
plt.ylabel('Price in Dollars')
plt.xlabel('Year')
plt.title('Meta')

# Customize x-axis ticks
plt.xticks(np.linspace(0, len(meta_df) - 1, num=num_quarters), [f'Q{q} {y}' for y in range(meta_df['Date'].dt.year.min(), meta_df['Date'].dt.year.max() + 1) for q in range(1, 5)])
# this might actually be wrong I just realised because our data starts from april

plt.show()

X_train, X_test, y_train, y_test = train_test_split(meta_df[['50_SMA']], meta_df[['Adj Close']], test_size=.2, shuffle=False)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Model Coefficients:", lr.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error (MAPE):", np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

plt.figure(figsize = (20, 10))
meta_df['50_SMA'].plot(color = 'g', lw = 1)
plt.plot(y_test.index, y_pred, color = 'b', lw = 1)
meta_df['Adj Close'].plot(figsize = (15, 8), fontsize = 12)
plt.title('Meta')

# Customize x-axis ticks
plt.xticks(np.linspace(0, len(meta_df) - 1, num=num_quarters), [f'Q{q} {y}' for y in range(meta_df['Date'].dt.year.min(), meta_df['Date'].dt.year.max() + 1) for q in range(1, 5)])
# this might actually be wrong I just realised because our data starts from april

plt.grid()
plt.show()