import numpy as np
import pandas as pd

# Simulate CPU and memory usage
np.random.seed(42)

cpu = np.random.normal(50, 10, 500)
memory = np.random.normal(60, 15, 500)

# Create dataframe
data = pd.DataFrame({
    "cpu": cpu,
    "memory": memory
})

print(data.head())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare sequence
X, y = [], []
for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i])
    y.append(scaled_data[i][0])  # predict CPU

X, y = np.array(X), np.array(y)

# Model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32)

# Prediction
pred = model.predict(X)

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_dim = X.shape[2]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data
autoencoder.fit(data, data, epochs=10, batch_size=32)

# Reconstruction error
recon = autoencoder.predict(data)
error = np.mean(np.power(data - recon, 2), axis=1)

# Threshold
threshold = np.mean(error) + 2*np.std(error)

anomalies = error > threshold
print("Anomalies detected:", sum(anomalies))

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=5, min_samples=5).fit(data)
labels = clustering.labels_

# -1 = anomaly
dbscan_anomalies = labels == -1

states = ["normal", "high_cpu", "failure"]
actions = ["do_nothing", "restart", "scale"]

# Q-table
Q = np.zeros((len(states), len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

def get_state(cpu):
    if cpu < 60:
        return 0  # normal
    elif cpu < 80:
        return 1  # high_cpu
    else:
        return 2  # failuredef get_reward(state, action):
    if state == 2 and action == 1:  # restart on failure
        return 10
    elif state == 1 and action == 2:  # scale on high load
        return 5
    elif action == 0:
        return -1
    else:
        return -5

for episode in range(1000):
    for i in range(len(cpu)):
        state = get_state(cpu[i])

        # epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(len(actions))
        else:
            action = np.argmax(Q[state])

        reward = get_reward(state, action)

        next_state = get_state(cpu[i])

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

print("Q-table:\n", Q)

def take_action(action):
    if action == 0:
        print("No action taken")
    elif action == 1:
        print("Restarting container...")
    elif action == 2:
        print("Scaling system...")

for i in range(10):
    state = get_state(cpu[i])
    action = np.argmax(Q[state])
    take_action(action)

import matplotlib.pyplot as plt

plt.plot(cpu, label="CPU")
plt.plot(pred.flatten(), label="Predicted CPU")
plt.legend()
plt.title("CPU Prediction")
plt.show()


