# ml/train.py
import numpy as np
import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import neptune
from ml.data import load_data


def train_model():
   # Neptune initialization
   run = neptune.init_project(
       project="dylanad2/CS222",
       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxY2EzZGZhZS1mMjNlLTQzMGYtOWI3NC1jMTE5OTQzYmQzZTAifQ=="
   )


   # Load and preprocess data
   df = load_data()
   df = df[['Close']]
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(df)


   # Prepare training data
   def create_dataset(dataset, time_step=1):
       X, Y = [], []
       for i in range(len(dataset)-time_step-1):
           a = dataset[i:(i+time_step), 0]
           X.append(a)
           Y.append(dataset[i + time_step, 0])
       return np.array(X), np.array(Y)


   time_step = 60
   X, y = create_dataset(scaled_data, time_step)
   X = X.reshape(X.shape[0], X.shape[1], 1)


   # Split into training and testing
   train_size = int(len(X) * 0.8)
   X_train, X_test = X[:train_size], X[train_size:]
   y_train, y_test = y[:train_size], y[train_size:]
   
   # Build the LSTM model
   model = Sequential([
       LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
       LSTM(50, return_sequences=False),
       Dense(25),
       Dense(1)
   ])
   model.compile(optimizer='adam', loss='mean_squared_error')


   # Train the model
   history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)


   # Log training metrics to Neptune
   for epoch, loss in enumerate(history.history['loss']):
       run[f'train/loss'].log(loss, epoch)


   # Evaluate the model
   test_loss = model.evaluate(X_test, y_test)
   run['eval/test_loss'] = test_loss


   # Save the model
   model.save('myapp/ml_model/lstm_model.h5')


   # Stop Neptune run
   run.stop()


   return test_loss



