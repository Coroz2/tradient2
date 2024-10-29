# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
# os.makedirs('models', exist_ok=True)
# import neptune.new as neptune
import neptune
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from supabase_client import save_predictions_to_supabase

class StockPredictor:
    def __init__(self, 
                 ticker_symbol,
                 neptune_project,
                 neptune_api_token,
                 historical_period="2y",
                 test_ratio=0.2,
                 window_size=50,
                 lstm_units=50,
                 optimizer="adam",
                 epochs=15,
                 batch_size=20,
                 run_name=None):
        """
        Initialize the StockPredictor with configurable parameters
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            neptune_project (str): Neptune project name (workspace/project_name)
            neptune_api_token (str): Neptune API token
            historical_period (str): Period for historical data (e.g., "2y", "1y", "6mo")
            test_ratio (float): Ratio of data to use for testing (0-1)
            window_size (int): Number of days to use for prediction window
            lstm_units (int): Number of LSTM units per layer
            optimizer (str): Optimizer for the LSTM model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            run_name (str, optional): Custom name for the Neptune run. 
                                     Defaults to f'LSTM_{ticker_symbol}'
        """
        self.ticker_symbol = ticker_symbol
        self.neptune_project = neptune_project
        self.neptune_api_token = neptune_api_token
        self.historical_period = historical_period
        self.test_ratio = test_ratio
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_name = run_name or f'LSTM_{ticker_symbol}'
        
        # Add these lines to clear any existing TensorFlow session
        tf.keras.backend.clear_session()
        
        # Initialize empty attributes
        self.model = None
        self.scaler = None
        self.run = None
        self.train = None
        self.test = None
        self.stockprices = None

    def fetch_data(self):
        """Fetch historical stock data"""
        ticker = yf.Ticker(self.ticker_symbol)
        historical_data = ticker.history(period=self.historical_period)
        self.stockprices = historical_data[['Close']].copy()
        self.stockprices.index = pd.to_datetime(self.stockprices.index)
        return self.stockprices

    def prepare_data(self):
        """Prepare and split the data into training and testing sets"""
        train_size = int((1 - self.test_ratio) * len(self.stockprices))
        self.train = self.stockprices[:train_size][['Close']]
        self.test = self.stockprices[train_size:][['Close']]
        
        # Scale the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.stockprices[["Close"]])
        scaled_data_train = scaled_data[:train_size]
        
        # Prepare sequences
        X_train, y_train = self._extract_seqX_outcomeY(scaled_data_train, 
                                                      self.window_size, 
                                                      self.window_size)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        return X_train, y_train

    def initialize_neptune(self):
        """Initialize Neptune.ai run"""
        self.run = neptune.init_run(
            project=self.neptune_project,
            api_token=self.neptune_api_token,
            name=self.run_name,
            description='stock-prediction-machine-learning',
            tags=['stockprediction', 'LSTM', 'neptune']
        )
        
        # Log hyperparameters
        self.run["LSTM_args"] = {
            "units": self.lstm_units,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "window_size": self.window_size
        }

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        # Clear any existing model first
        if self.model is not None:
            del self.model
            tf.keras.backend.clear_session()
        
        # Build the model
        inp = Input(shape=(input_shape[1], 1))
        x = LSTM(units=self.lstm_units, return_sequences=True)(inp)
        x = LSTM(units=self.lstm_units)(x)
        out = Dense(1, activation="linear")(x)
        self.model = Model(inp, out)
        self.model.compile(loss="mean_squared_error", optimizer=self.optimizer)
        return self.model

    def train_model(self, X_train, y_train):
        """Train the LSTM model"""
        # Add callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Train the model and log metrics manually
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_split=0.1,
            shuffle=True,
            callbacks=callbacks
        )
        
        # Log training history to Neptune
        for epoch in range(len(history.history['loss'])):
            self.run["metrics/epoch/loss"].log(history.history['loss'][epoch])
            self.run["metrics/epoch/val_loss"].log(history.history['val_loss'][epoch])
        
        return history

    def make_predictions(self):
        """Make predictions on the test set"""
        X_test = self._preprocess_testdat()
        predicted_price_ = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price_)
        self.test["Predictions_lstm"] = predicted_price
        return self.test

    def evaluate_model(self):
        """Evaluate the model performance"""
        rmse = self._calculate_rmse(
            np.array(self.test["Close"]),
            np.array(self.test["Predictions_lstm"])
        )
        mape = self._calculate_mape(
            np.array(self.test["Close"]),
            np.array(self.test["Predictions_lstm"])
        )
        
        self.run["RMSE"] = rmse
        self.run["MAPE (%)"] = mape
        return rmse, mape

    def save_model(self, save_dir='models'):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f'lstm_{self.ticker_symbol}.h5')
        self.model.save(model_save_path)
        self.run['system/ml/models'].upload(model_save_path)

        # Save both training and test data to Supabase
        save_predictions_to_supabase(
            self.train, 
            self.test, 
            self.test["Predictions_lstm"], 
            self.ticker_symbol
        )
        print("Saved predictions to supabase")
        return model_save_path

    # Helper methods (private)
    def _extract_seqX_outcomeY(self, data, N, offset):
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

    def _calculate_rmse(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def _calculate_mape(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

    def _preprocess_testdat(self):
        raw = self.stockprices["Close"][len(self.stockprices) - len(self.test) - self.window_size:].values
        raw = raw.reshape(-1,1)
        raw = self.scaler.transform(raw)

        X_test = [raw[i-self.window_size:i, 0] for i in range(self.window_size, raw.shape[0])]
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test

    def plot_predictions(self):
        """Plot the predictions vs actual values"""
        fig = plt.figure(figsize=(14,7))
        plt.plot(self.train.index, self.train["Close"], label="Train Closing Price")
        plt.plot(self.test.index, self.test["Close"], label="Test Closing Price")
        plt.plot(self.test.index, self.test["Predictions_lstm"], label="Predicted Closing Price")
        plt.title("LSTM Model")
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend(loc="upper left")
        self.run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))
        return fig

# Example usage
predictor = StockPredictor(
    ticker_symbol="PLTR",
    neptune_project="dylanad2/CS222",
    neptune_api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxY2EzZGZhZS1mMjNlLTQzMGYtOWI3NC1jMTE5OTQzYmQzZTAifQ==",
    run_name="PLTR_class_based_test_run", 
    historical_period="2y",
    test_ratio=0.2,
    window_size=50,
    lstm_units=50,
    optimizer="adam",
    epochs=15,
    batch_size=20
)

# Run the full prediction pipeline
predictor.fetch_data()
X_train, y_train = predictor.prepare_data()
predictor.initialize_neptune()
predictor.build_model(X_train.shape)
predictor.train_model(X_train, y_train)
predictions = predictor.make_predictions()
rmse, mape = predictor.evaluate_model()
predictor.plot_predictions()
model_path = predictor.save_model()
predictor.run.stop()
