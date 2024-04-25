# combined model of market sentiment, moving averages, and trading volumes for MSFT stock
# goal: weighted average of all three predictions 

# Import necessary functions from other modules
from market_sentiment.microsoftmarketsentiment import predictions as sentiment_prediction
from moving_averages.msft_moving_averages import y_pred as ma_prediction
from trading_volumes.microsofttradingvolume import get_predictions as get_volume_prediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def combine_predictions(ma_prediction, volume_prediction, sentiment_prediction):
    """
    Combine predictions from three models using dynamically determined weights.

    Args:
    - ma_prediction (numpy array): Predictions from the moving averages model.
    - volume_prediction (numpy array): Predictions from the trading volumes model.
    - sentiment_prediction (numpy array): Predictions from the market sentiment model.

    Returns:
    - combined_prediction (numpy array): Weighted average of the predictions.
    """

    volume_prediction = volume_prediction[:50]
    sentiment_prediction = sentiment_prediction[:50]
    ma_prediction = ma_prediction[:50]


    # Calculate inverse of mean squared error for each model as a proxy for performance
    ma_error = 1 / np.mean((ma_prediction) ** 2)
    volume_error = 1 / np.mean((volume_prediction) ** 2)
    sentiment_error = 1 / np.mean((sentiment_prediction) ** 2)

    # Normalize errors to get weights
    total_error = ma_error + volume_error + sentiment_error
    ma_weight = ma_error / total_error
    volume_weight = volume_error / total_error
    sentiment_weight = sentiment_error / total_error

    best_weights = [ma_weight, volume_weight, sentiment_weight]

    # Combine predictions using dynamically determined weights
    combined_prediction = (ma_weight * ma_prediction +
                           volume_weight * volume_prediction +
                           sentiment_weight * sentiment_prediction)

    return best_weights, combined_prediction

def calculate_mape(actual_prices, predicted_prices):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between actual and predicted prices.

    Args:
    - actual_prices (numpy array): Array containing the actual stock prices.
    - predicted_prices (numpy array): Array containing the predicted stock prices.

    Returns:
    - mape (float): Mean Absolute Percentage Error (MAPE) between actual and predicted prices.
    """
    absolute_errors = np.abs(actual_prices - predicted_prices)
    percentage_errors = absolute_errors / actual_prices * 100
    mape = np.mean(percentage_errors)
    return mape


def main():
    # Get predictions from other modules
    volume_prediction = get_volume_prediction()

    # Combine predictions and find optimal weights
    best_weights, combined_prediction = combine_predictions(ma_prediction, volume_prediction, sentiment_prediction)

    # Load the NVIDIA stock data from the CSV file
    msft_data = pd.read_csv("data/MSFT.csv")

    # Convert the 'Date' column to datetime format
    msft_data['Date'] = pd.to_datetime(msft_data['Date'])

    # Filter the data to get the first 50 days of 2024
    msft_data_2024 = msft_data[msft_data['Date'].dt.year == 2024].head(50)

    # Extract actual stock prices for the first 50 days of 2024
    actual_prices = msft_data_2024['Adj Close'].values

    # Generate time series for the first 50 days of 2024
    time_series = msft_data_2024['Date'].values

    # Calculate the average prediction along the columns axis
    average_prediction = np.mean(combined_prediction, axis=1)

    mape = calculate_mape(actual_prices, average_prediction)

    # Print the results
    print("Best weights:", best_weights)
    print("Combined Prediction:", combined_prediction)
    print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

    # Plot actual vs predicted stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, actual_prices, label='Actual Prices', color='blue')
    plt.plot(time_series, average_prediction, label='Predicted Prices', color='red')

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices for Microsoft in 2024')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
