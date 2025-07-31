import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
import random
np.random.seed(123)
tf.random.set_seed(123)
random.seed(123)

df = pd.read_csv('stock_data.csv', index_col=0)
prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

def make_sequences(data, lookback=45):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

lookback = 45
X, y = make_sequences(scaled_prices, lookback)
X = X[..., np.newaxis]

split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, input_shape=(lookback, 1), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, callbacks=[es], verbose=2)

y_pred = model.predict(X_test).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f"Test MAE: {mae:.2f}")

model.save('lstm_model.keras')

plt.figure(figsize=(12,5))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred_actual, label='Predicted', alpha=0.7)
plt.title('Stock Price Prediction Model')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('plot_alt_lstm.png')
plt.show()
