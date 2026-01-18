import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib  # Added for professional scaler persistence
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set plot style
plt.style.use('fivethirtyeight')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# 1. Load Dataset
df = load_data('Toyota_Data.csv')

# 2. Data Preparation & Scaling
data = df[['Close']].values
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Fit Scaler ONLY on Training Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# --- CRITICAL UPDATE: SAVE THE SCALER ---
joblib.dump(scaler, 'scaler.gz') 
print("✅ Scaler saved as 'scaler.gz'")

# 3. Sequence Creation
seq_length = 60
X_train, y_train = create_sequences(scaled_train, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 4. Model Architecture
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Training
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# 6. Evaluation & Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(y_test_unscaled, predictions))
mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100
accuracy = 100 - mape

print(f"\n----- Final Performance Metrics -----")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Overall Model Accuracy: {accuracy:.2f}%")

# 7. Model Persistence
model.save('toyota_model.h5')
print("✅ Model saved as 'toyota_model.h5'")