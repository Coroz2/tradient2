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
from .supabase_client import save_predictions_to_supabase
from dotenv import load_dotenv
from .overfitting_detector import OverfittingDetector
import time

load_dotenv()

class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            print(f"\nTimeout reached after {self.seconds} seconds")

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
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            historical_data = ticker.history(period=self.historical_period)
            
            if historical_data.empty:
                raise ValueError(f"No data found for {self.ticker_symbol}. The stock might be delisted.")
            
            # Replace deprecated fillna method
            historical_data = historical_data.ffill().bfill()
            
            self.stockprices = historical_data[['Close']].copy()
            self.stockprices.index = pd.to_datetime(self.stockprices.index)
            
            # Verify we have enough data
            if len(self.stockprices) < self.window_size * 2:
                raise ValueError(
                    f"Insufficient data for {self.ticker_symbol}. "
                    f"Found only {len(self.stockprices)} days of data, but need at least {self.window_size * 2} days."
                )
            
            return self.stockprices
        
        except Exception as e:
            if "No data found" in str(e):
                raise ValueError(f"Stock {self.ticker_symbol} appears to be delisted or invalid.")
            raise

    def prepare_data(self):
        """Prepare and split the data into training and testing sets"""
        if self.stockprices is None or len(self.stockprices) == 0:
            raise ValueError("No stock data available. Please fetch data first.")
            
        # Calculate technical indicators
        feature_columns = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Price_Change', 'Price_Change_MA5']
        
        # Calculate all indicators
        self.stockprices['MA20'] = self.stockprices['Close'].rolling(window=20).mean()
        self.stockprices['MA50'] = self.stockprices['Close'].rolling(window=50).mean()
        self.stockprices['RSI'] = self._calculate_rsi(self.stockprices['Close'])
        self.stockprices['MACD'] = self._calculate_macd(self.stockprices['Close'])
        self.stockprices['Price_Change'] = self.stockprices['Close'].pct_change()
        self.stockprices['Price_Change_MA5'] = self.stockprices['Price_Change'].rolling(window=5).mean()
        
        # Forward fill and standardize all features
        self.stockprices[feature_columns] = self.stockprices[feature_columns].fillna(method='ffill').fillna(0)
        
        # Split data
        # Calculate technical indicators
        feature_columns = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Price_Change', 'Price_Change_MA5']
        
        # Calculate all indicators
        self.stockprices['MA20'] = self.stockprices['Close'].rolling(window=20).mean()
        self.stockprices['MA50'] = self.stockprices['Close'].rolling(window=50).mean()
        self.stockprices['RSI'] = self._calculate_rsi(self.stockprices['Close'])
        self.stockprices['MACD'] = self._calculate_macd(self.stockprices['Close'])
        self.stockprices['Price_Change'] = self.stockprices['Close'].pct_change()
        self.stockprices['Price_Change_MA5'] = self.stockprices['Price_Change'].rolling(window=5).mean()
        
        # Forward fill and standardize all features
        self.stockprices[feature_columns] = self.stockprices[feature_columns].fillna(method='ffill').fillna(0)
        
        # Split data
        train_size = int((1 - self.test_ratio) * len(self.stockprices))
        
        # Scale features
        # Scale features
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.stockprices[feature_columns])
        scaled_data = self.scaler.fit_transform(self.stockprices[feature_columns])
        
        # Prepare training data with all features
        # Prepare training data with all features
        scaled_data_train = scaled_data[:train_size]
        X_train, y_train = [], []
        
        for i in range(self.window_size, len(scaled_data_train)):
            X_train.append(scaled_data_train[i-self.window_size:i])
            y_train.append(scaled_data_train[i, 0])  # Only predict Close price
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.train = self.stockprices[:train_size][['Close']]
        self.test = self.stockprices[train_size:][['Close']]
        X_train, y_train = [], []
        
        for i in range(self.window_size, len(scaled_data_train)):
            X_train.append(scaled_data_train[i-self.window_size:i])
            y_train.append(scaled_data_train[i, 0])  # Only predict Close price
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.train = self.stockprices[:train_size][['Close']]
        self.test = self.stockprices[train_size:][['Close']]
        
        return X_train, y_train

    def initialize_neptune(self, groupTags):
        """Initialize Neptune.ai run"""
        tags = ['LSTM']
        self.run = neptune.init_run(
            project=self.neptune_project,
            api_token=self.neptune_api_token,
            name=self.run_name,
            description='stock-prediction-machine-learning',
            tags=tags
            
        )
        self.run["sys/group_tags"].add(groupTags)
        
        # Log hyperparameters
        self.run["LSTM_args"] = {
            "units": self.lstm_units,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "window_size": self.window_size
        }

    def build_model(self, input_shape):
        """Build and compile the LSTM model with regularization"""
        """Build and compile the LSTM model with regularization"""
        if self.model is not None:
            del self.model
            tf.keras.backend.clear_session()
        
        # Build the model with regularization
        inp = Input(shape=(input_shape[1], input_shape[2]))  # Changed to handle multiple features
        
        # First LSTM layer
        x = LSTM(units=self.lstm_units, 
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Second LSTM layer
        x = LSTM(units=self.lstm_units//2,
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third LSTM layer
        x = LSTM(units=self.lstm_units//4,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        # Build the model with regularization
        inp = Input(shape=(input_shape[1], input_shape[2]))  # Changed to handle multiple features
        
        # First LSTM layer
        x = LSTM(units=self.lstm_units, 
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Second LSTM layer
        x = LSTM(units=self.lstm_units//2,
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third LSTM layer
        x = LSTM(units=self.lstm_units//4,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = Dense(1, activation="linear")(x)
        
        
        self.model = Model(inp, out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss="huber", optimizer=optimizer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss="huber", optimizer=optimizer)
        return self.model

    def train_model(self, X_train, y_train):
        """Train the LSTM model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                patience=10,
                restore_best_weights=True,
                min_delta=0.0001
                min_delta=0.0001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                min_delta=0.0001
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                min_delta=0.0001
            ),
            TimeoutCallback(seconds=600)
            TimeoutCallback(seconds=600)
        ]
        
        # Split data into train and validation sets
        val_split = 0.15
        split_idx = int(len(X_train) * (1 - val_split))
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        
        try:
            # Train the model with simplified parameters
            history = self.model.fit(
                X_train_split, y_train_split,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                validation_data=(X_val, y_val),
                shuffle=True,
                callbacks=callbacks
            )
            
            # Create and use the OverfittingDetector
            try:
                detector = OverfittingDetector(
                    history=history,
                    model=self.model,
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_val=X_val,
                    y_val=y_val,
                    neptune_run=self.run
                )
                
                is_overfit, metrics = detector.analyze_overfitting(threshold=0.1)
                
                # Enhanced logging of results
                print("\n=== Overfitting Analysis Results ===")
                print(f"Is model overfitting? {'Yes' if is_overfit else 'No'}")
                print("\nMetrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
                # Log training history to Neptune
                for epoch in range(len(history.history['loss'])):
                    self.run["metrics/epoch/loss"].log(history.history['loss'][epoch])
                    self.run["metrics/epoch/val_loss"].log(history.history['val_loss'][epoch])
                    
            except Exception as e:
                print(f"\nOverfitting analysis error: {str(e)}")
                print("Training completed but overfitting analysis failed.")
            
            return history
            
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            raise

    def make_predictions(self):
        """Make predictions on the test set"""
        X_test = self._preprocess_testdat()
        predicted_price_ = self.model.predict(X_test)
        
        # Create a dummy array with same shape as training data
        dummy = np.zeros((predicted_price_.shape[0], 7))  # 7 is number of features
        dummy[:, 0] = predicted_price_.flatten()  # Put predictions in first column (Close price)
        
        # Inverse transform
        predicted_price = self.scaler.inverse_transform(dummy)[:, 0]  # Take only the Close price column
        
        # Ensure we're only using available test data
        current_date = pd.Timestamp.now()  # timezone-naive
        self.test.index = self.test.index.tz_localize(None)  # Remove timezone information
        self.test = self.test[self.test.index <= current_date]
        self.test["Predictions_lstm"] = predicted_price[:len(self.test)]
        
        print(f"Debug: Test data date range: {self.test.index[0]} to {self.test.index[-1]}")
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
        """Save the trained model and predictions"""
        """Save the trained model and predictions"""
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f'lstm_{self.ticker_symbol}.h5')
        self.model.save(model_save_path)
        self.run['system/ml/models'].upload(model_save_path)

        # Debug logging
        print("\nDEBUG: Verification of predictions")
        print("Test Data Shape:", self.test.shape)
        print("\nSample of Test Data and Predictions:")
        debug_df = pd.DataFrame({
            'Date': self.test.index[-5:],  # Last 5 dates
            'Actual': self.test['Close'][-5:],
            'Predicted': self.test['Predictions_lstm'][-5:],
            'Difference': self.test['Close'][-5:] - self.test['Predictions_lstm'][-5:]
        })
        print(debug_df.to_string())
        
        # Prepare data for saving
        predictions_df = pd.DataFrame({
            'Date': self.test.index,
            'Close': self.test['Close'],
            'Predictions_lstm': self.test['Predictions_lstm']
        })
        

        # Debug logging
        print("\nDEBUG: Verification of predictions")
        print("Test Data Shape:", self.test.shape)
        print("\nSample of Test Data and Predictions:")
        debug_df = pd.DataFrame({
            'Date': self.test.index[-5:],  # Last 5 dates
            'Actual': self.test['Close'][-5:],
            'Predicted': self.test['Predictions_lstm'][-5:],
            'Difference': self.test['Close'][-5:] - self.test['Predictions_lstm'][-5:]
        })
        print(debug_df.to_string())
        
        # Prepare data for saving
        predictions_df = pd.DataFrame({
            'Date': self.test.index,
            'Close': self.test['Close'],
            'Predictions_lstm': self.test['Predictions_lstm']
        })
        
        # Save both training and test data to Supabase
        save_predictions_to_supabase(
            self.train, 
            predictions_df,
            predictions_df['Predictions_lstm'],
            predictions_df,
            predictions_df['Predictions_lstm'],
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
        """Preprocess test data with all features"""
        feature_columns = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Price_Change', 'Price_Change_MA5']
        raw = self.stockprices[feature_columns][len(self.stockprices) - len(self.test) - self.window_size:].values
        """Preprocess test data with all features"""
        feature_columns = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Price_Change', 'Price_Change_MA5']
        raw = self.stockprices[feature_columns][len(self.stockprices) - len(self.test) - self.window_size:].values
        raw = self.scaler.transform(raw)
        
        X_test = []
        for i in range(self.window_size, raw.shape[0]):
            X_test.append(raw[i-self.window_size:i])
        
        X_test = []
        for i in range(self.window_size, raw.shape[0]):
            X_test.append(raw[i-self.window_size:i])
        X_test = np.array(X_test)
        
        
        return X_test

    def plot_predictions(self):
        """Plot the predictions vs actual values"""
        fig = plt.figure(figsize=(14,7))
        # Plot training data
        plt.plot(self.train.index, self.train["Close"], 
                 label="Training Data", color='blue', alpha=0.6)
        # Plot test data (actual)
        plt.plot(self.test.index, self.test["Close"], 
                 label="Actual Prices", color='green', alpha=0.8)
        # Plot predictions
        plt.plot(self.test.index, self.test["Predictions_lstm"], 
                 label="Predicted Prices", color='red', linestyle='--')
        
        plt.title(f"Stock Price Prediction - {self.ticker_symbol}")
        # Plot training data
        plt.plot(self.train.index, self.train["Close"], 
                 label="Training Data", color='blue', alpha=0.6)
        # Plot test data (actual)
        plt.plot(self.test.index, self.test["Close"], 
                 label="Actual Prices", color='green', alpha=0.8)
        # Plot predictions
        plt.plot(self.test.index, self.test["Predictions_lstm"], 
                 label="Predicted Prices", color='red', linestyle='--')
        
        plt.title(f"Stock Price Prediction - {self.ticker_symbol}")
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Add confidence intervals
        std_dev = np.std(self.test["Close"] - self.test["Predictions_lstm"])
        plt.fill_between(self.test.index, 
                        self.test["Predictions_lstm"] - 2*std_dev,
                        self.test["Predictions_lstm"] + 2*std_dev,
                        color='red', alpha=0.1)
        
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Add confidence intervals
        std_dev = np.std(self.test["Close"] - self.test["Predictions_lstm"])
        plt.fill_between(self.test.index, 
                        self.test["Predictions_lstm"] - 2*std_dev,
                        self.test["Predictions_lstm"] + 2*std_dev,
                        color='red', alpha=0.1)
        
        self.run["Plot of Stock Predictions"].upload(neptune.types.File.as_image(fig))
        return fig

    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd.ewm(span=signal, adjust=False).mean()

    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd.ewm(span=signal, adjust=False).mean()

# Example usage
# predictor = StockPredictor(
#     ticker_symbol="PLTR",
#     neptune_project=os.getenv('NEPTUNE_PROJECT'),
#     neptune_api_token=os.getenv('NEPTUNE_API_TOKEN'),
#     run_name="PLTR_class_based_test_run", 
#     historical_period="2y",
#     test_ratio=0.2,
#     window_size=50,
#     lstm_units=50,
#     optimizer="adam",
#     epochs=15,
#     batch_size=20
# )

# # Run the full prediction pipeline
# predictor.fetch_data()
# X_train, y_train = predictor.prepare_data()
# predictor.initialize_neptune()
# predictor.build_model(X_train.shape)
# predictor.train_model(X_train, y_train)
# predictions = predictor.make_predictions()
# rmse, mape = predictor.evaluate_model()
# predictor.plot_predictions()
# model_path = predictor.save_model()
# predictor.run.stop()
