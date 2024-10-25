
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# import neptune.new as neptune
import neptune
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


# Collecting the data using yfinance
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(period="2y")  # Using 2 years of data for better results
print("Historical Data:")
print(historical_data.head())

# Prepare the data
stockprices = historical_data[['Close']].copy()
stockprices.index = pd.to_datetime(stockprices.index)

# Split data into train and test sets
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = len(stockprices) - train_size

print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size][['Close']]
test = stockprices[train_size:][['Close']]

# Define helper functions
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N : i])
        y.append(data[i])

    return np.array(X), np.array(y)

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def calculate_perf_metrics(var, stockprices, train_size, run):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )

    ## Log to Neptune
    run["RMSE"] = rmse
    run["MAPE (%)"] = mape

    return rmse, mape

def plot_stock_trend(var, cur_title, stockprices, run):
    plt.figure(figsize=(14, 7))
    ax = stockprices[["Close", var]].plot(figsize=(14, 7))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

    ## Log to Neptune
    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )

# Setting up Neptune.ai project
# Replace 'your_workspace/your_project_name' with your Neptune project name
# and 'YOUR_NEPTUNE_API_TOKEN' with your actual Neptune API token

run = neptune.init_run(
    project="dylanad2/CS222",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxY2EzZGZhZS1mMjNlLTQzMGYtOWI3NC1jMTE5OTQzYmQzZTAifQ==",
    name='LSTM',
    description='stock-prediction-machine-learning',
    tags=['stockprediction', 'LSTM', 'neptune'],
)

# Log the LSTM hyperparameters
layer_units = 50
optimizer = "adam"
cur_epochs = 15
cur_batch_size = 20
window_size = 50  # Using a window size of 50 days

cur_LSTM_args = {
    "units": layer_units,
    "optimizer": optimizer,
    "batch_size": cur_batch_size,
    "epochs": cur_epochs,
    "window_size": window_size
}

run["LSTM_args"] = cur_LSTM_args

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["Close"]])
scaled_data_train = scaled_data[: train.shape[0]]

# Extract sequences for training
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)

# Reshape X_train for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Setup Neptune's Keras integration
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

neptune_callback = NeptuneCallback(run=run)

# Build the LSTM model
def Run_LSTM(X_train, layer_units=50):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model

model = Run_LSTM(X_train, layer_units=layer_units)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=cur_epochs,
    batch_size=cur_batch_size,
    verbose=1,
    validation_split=0.1,
    shuffle=True,
    callbacks=[neptune_callback],
)

# Prepare test data
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
    raw = data["Close"][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

X_test = preprocess_testdat()

# Make predictions on the test set
predicted_price_ = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_)

# Compare with actual test prices
test["Predictions_lstm"] = predicted_price

# Evaluate performance
rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
mape_lstm = calculate_mape(np.array(test["Close"]), np.array(test["Predictions_lstm"]))

# Log metrics to Neptune
run["RMSE"] = rmse_lstm
run["MAPE (%)"] = mape_lstm

# Plot the predictions vs actual values
def plot_stock_trend_lstm(train, test, run):
    fig = plt.figure(figsize = (14,7))
    plt.plot(train.index, train["Close"], label = "Train Closing Price")
    plt.plot(test.index, test["Close"], label = "Test Closing Price")
    plt.plot(test.index, test["Predictions_lstm"], label = "Predicted Closing Price")
    plt.title("LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")

    ## Log image to Neptune
    run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))

plot_stock_trend_lstm(train, test, run)

# Stop the Neptune run
run.stop()
