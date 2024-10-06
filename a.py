import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import warnings



# Suppress all warnings
warnings.filterwarnings("ignore")
# Load your CSV files
top_10_products = pd.read_csv('top_10_products.csv')
merged_df = pd.read_csv('merged_df.csv')



st.title("Demand Forecasting ")



# Function to calculate error
def calculate_error(actual, predicted):
    return actual - predicted

# Time-based cross-validation
def time_based_cross_validation(data, n_splits=5):
    fold_size = len(data) // n_splits
    for i in range(n_splits):
        train = data[:(i + 1) * fold_size]
        test = data[(i + 1) * fold_size:(i + 2) * fold_size] if (i + 1) * fold_size < len(data) else data[(i + 1) * fold_size:]
        yield train, test



# Add this function to forecast and plot results
def forecast_and_plot(stock_code):
    # (Your original code goes here, adapt as necessary)
    product_data = merged_df[merged_df['StockCode'] == stock_code].copy()
    # Step 2: Aggregate the data by date and sum quantities
    product_data['InvoiceDate'] = pd.to_datetime(product_data['InvoiceDate'], dayfirst=True)
    daily_data = product_data.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    daily_data.columns = ['ds', 'y']  # Rename columns for Prophet
    daily_data.set_index('ds', inplace=True)

    # Initialize variables for cross-validation
    best_rmse = float('inf')
    best_params = {}  # Dictionary to store best model parameters

    # Use time-based cross-validation to find the best parameters
    for train_data, test_data in time_based_cross_validation(daily_data, n_splits=5):

        # Reset index to ensure the columns are correct
        train_data.reset_index(inplace=True)



        # Create and fit the Prophet model
        model = Prophet()
        model.fit(train_data)
        if not test_data.empty:

            # Make predictions for the test set

            future = pd.DataFrame({'ds': test_data.index})
            forecast = model.predict(future)
            #print(forecast)

            forecast.set_index('ds', inplace=True)
            # forecast = model.predict(test_data)
            # Merge forecast with test data

            #print(forecast.info())
            forecasted_data = forecast['yhat'].copy()
            # If you want to convert it to a DataFrame later
            forecasted_data = forecasted_data.to_frame()
            #print(forecasted_data)


            # Reindex the forecasted data to match test data index
            #forecasted_yhat = forecasted_data['yhat'].reindex(test_data.index)

            # Calculate RMSE for the test set
            rmse = np.sqrt(mean_squared_error(test_data['y'], forecasted_data['yhat']))
            if rmse < best_rmse:
                best_rmse = rmse
                # Extract the parameters of the best model
                best_params = {
                    'yearly_seasonality': model.yearly_seasonality,
                    'weekly_seasonality': model.weekly_seasonality,
                    'seasonality_mode': model.seasonality_mode,
                    'holidays': model.holidays,
                    'changepoint_prior_scale': model.changepoint_prior_scale,
                    'seasonality_prior_scale': model.seasonality_prior_scale
                }
        else:
          print(f"Warning: Test data for StockCode {stock_code} is empty.")

    print(f"StockCode: {stock_code}, Best RMSE from cross-validation: {best_rmse}")

    # Get train and test splits
    train_size = int(len(daily_data) * 0.8)
    train_data = daily_data[:train_size]
    test_data = daily_data[train_size:]

    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    # print(train_data)
    # print(test_data)
    # Create and fit a new Prophet model with the best parameters
    final_model = Prophet(
        yearly_seasonality=best_params['yearly_seasonality'],
        weekly_seasonality=best_params['weekly_seasonality'],
        seasonality_mode=best_params['seasonality_mode'],
        holidays=best_params['holidays'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale']
    )
    final_model.fit(train_data)

    # Predictions on the train and test set
    train_forecast = final_model.predict(train_data[['ds']])
    test_forecast = final_model.predict(test_data[['ds']])

    # Ensure predictions are not negative
    train_forecast['yhat'] = np.maximum(train_forecast['yhat'], 0)  # Ensure predictions are not negative
    test_forecast['yhat'] = np.maximum(test_forecast['yhat'], 0)    # Ensure predictions are not negative

    # Determine the number of days to forecast (105 days)
    forecast_length = 105

    last_test_date = pd.to_datetime(test_data['ds'].max())

    # Generate date range starting from the next day of the last date in test_data, for 105 days
    future_dates = pd.date_range(start=last_test_date + pd.Timedelta(days=1), periods=105, freq='D')

    # Create a DataFrame with one column 'ds' containing the generated dates
    future_df = pd.DataFrame({'ds': future_dates})

    # Generate the forecast for the entire period
    forecast = final_model.predict(future_df)

    # Ensure predictions are not negative
    forecast['yhat'] = np.maximum(forecast['yhat'], 0)  # Set negative values to 0

    # Calculate RMSE and MAE for the test set
    test_rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
    test_mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])

    print(f"StockCode: {stock_code}, Test RMSE: {test_rmse}, Test MAE: {test_mae}")

    # Plot train actual, train predicted, test actual, test predicted, and future forecast
    plt.figure(figsize=(14, 8))

    # Plot train actual
    plt.plot(train_data['ds'], train_data['y'], label='Train Actual', color='blue', linestyle='--')

    # Plot train predicted
    plt.plot(train_forecast['ds'], train_forecast['yhat'], label='Train Predicted', color='green')

    # Plot test actual
    plt.plot(test_data['ds'], test_data['y'], label='Test Actual', color='orange', linestyle='--')

    # Plot test predicted
    plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Test Predicted', color='red')

    # Plot future forecast
    plt.plot(future_df['ds'], forecast['yhat'], label='15-week Forecast', color='purple')

    # Add titles and legend
    plt.title(f'StockCode: {stock_code} - Train/Test Actual vs Predicted and 15-week Forecast')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    st.pyplot(plt)

    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    # Calculate errors for train and test
    train_error = calculate_error(train_data['y'], train_forecast['yhat'])
    test_error = calculate_error(test_data['y'], test_forecast['yhat'])

    # Plot histogram of train and test errors
    plt.figure(figsize=(12, 6))

    # Train error histogram with KDE
    plt.subplot(121)
    sns.histplot(train_error, bins=20, color='orange', edgecolor='orange', kde=True)
    plt.title(f'StockCode: {stock_code} - Train Error Distribution')

    # Test error histogram with KDE
    plt.subplot(122)
    sns.histplot(test_error, bins=20, color='green', edgecolor='green', kde=True)
    plt.title(f'StockCode: {stock_code} - Test Error Distribution')

    plt.tight_layout()
    st.pyplot(plt)


# Dropdown menu for selecting products with a placeholder option
selected_product = st.selectbox("Select a Product", top_10_products['StockCode'])



if selected_product:
    # Call your forecasting and plotting code here
    forecast_and_plot(selected_product)
