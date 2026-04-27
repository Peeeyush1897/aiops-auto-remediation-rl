# ==============================
# AIOps LSTM + Autoencoder Model
# ==============================

import numpy as np
import pandas as pd

# ------------------------------
# Step 1: Simulate system metrics
# ------------------------------
np.random.seed(42)

cpu = np.random.normal(50, 10, 500)
memory = np.random.normal(60, 15, 500)

data = pd.DataFrame({
    "cpu": cpu,
    "memory": memory
})

print("Sample Data:")
print(data.head())


# ------------------------------
# Step 2: Preprocessing
# ------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# ------------------------------
# Step 3: Prepare sequences (LSTM)
# ------------------------------
X, y = [], []

for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i])
    y.append(scaled_data[i][0])  # Predict CPU

X = np.array(X)
y = np.array(y)


# ------------------------------
# Step 4: LSTM Model
# ------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("\nTraining LSTM model...")
model.fit(X, y, epochs=5, batch_size=32)


# ------------------------------
# Step 5: Prediction
# ------------------------------
predictions = model.predict(X)
print("\nPrediction sample:")
print(predictions[:5])


# ------------------------------
# Step 6: Autoencoder for anomaly detection
# ------------------------------
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_dim = scaled_data.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("\nTraining Autoencoder...")
autoencoder.fit(scaled_data, scaled_data, epochs=10, batch_size=32)


# ------------------------------
# Step 7: Reconstruction Error (Anomaly Score)
# ------------------------------
reconstructed = autoencoder.predict(scaled_data)
mse = np.mean(np.power(scaled_data - reconstructed, 2), axis=1)

threshold = np.mean(mse) + 2 * np.std(mse)

print("\nAnomaly Threshold:", threshold)

anomalies = mse > threshold
print("Number of anomalies detected:", np.sum(anomalies))
