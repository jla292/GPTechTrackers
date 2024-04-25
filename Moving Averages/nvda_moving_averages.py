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

# import the closing price data of NVidia stock for the period of 2 years -

nvda_df = pd.read_csv('data/NVDA.csv')
nvda_df.head()

# convert Date column to datetime
nvda_df['Date'] = pd.to_datetime(nvda_df['Date'], format = '%Y-%m-%d')
# sort by datetime
nvda_df.sort_values(by='Date', inplace=True, ascending=True)

# Create 50 days simple moving average column
nvda_df['50_SMA'] = nvda_df['Close'].rolling(window = 50, min_periods = 1).mean()
nvda_df.dropna()
nvda_df.head()

# visualize data

# Calculate the number of quarters in the data range
num_quarters = (nvda_df['Date'].dt.year.max() - nvda_df['Date'].dt.year.min() + 1) * 4

nvda_df['Adj Close'].plot(figsize = (15, 8), fontsize = 12)
plt.grid()
plt.ylabel('Price in Dollars')
plt.xlabel('Year')
plt.title('NVidia')
# Customize x-axis ticks
plt.xticks(np.linspace(0, len(nvda_df) - 1, num=num_quarters), [f'Q{q} {y}' for y in range(nvda_df['Date'].dt.year.min(), nvda_df['Date'].dt.year.max() + 1) for q in range(1, 5)])

plt.show()

X_train, X_test, y_train, y_test = train_test_split(nvda_df[['50_SMA']], nvda_df[['Adj Close']], test_size=.2, shuffle=False)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Model Coefficients:", lr.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error (MAPE):", np.mean(np.abs((y_test - y_pred) / y_test)) * 100)


plt.figure(figsize = (20, 10))
nvda_df['50_SMA'].plot(color = 'g', lw = 1)
plt.plot(y_test.index, y_pred, color = 'b', lw = 1)
nvda_df['Adj Close'].plot(figsize = (15, 8), fontsize = 12)

# Customize x-axis ticks
plt.xticks(np.linspace(0, len(nvda_df) - 1, num=num_quarters), [f'Q{q} {y}' for y in range(nvda_df['Date'].dt.year.min(), nvda_df['Date'].dt.year.max() + 1) for q in range(1, 5)])
plt.title('NVidia')
plt.grid()
plt.show()