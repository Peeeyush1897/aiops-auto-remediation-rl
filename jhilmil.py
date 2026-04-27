# =========================================================
# Intelligent Auto-Remediation in DevOps using RL (AIOps)
# FINAL ONE-FILE IMPLEMENTATION
# =========================================================

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================================
# 1. SIMULATE SYSTEM METRICS (CPU)
# =========================================================
np.random.seed(42)
cpu_data = np.random.normal(60, 15, 300)

# Normalize for LSTM
scaler = MinMaxScaler()
cpu_scaled = scaler.fit_transform(cpu_data.reshape(-1, 1))

# =========================================================
# 2. LSTM PREDICTION MODEL
# =========================================================
X, y = [], []
for i in range(10, len(cpu_scaled)):
    X.append(cpu_scaled[i-10:i])
    y.append(cpu_scaled[i])

X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

print("\nTraining LSTM model...\n")
model.fit(X, y, epochs=3, batch_size=32, verbose=0)

predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

# =========================================================
# 3. ANOMALY DETECTION
# =========================================================
def is_anomaly(cpu):
    return cpu > 80  # threshold

# =========================================================
# 4. RL (Q-LEARNING)
# =========================================================

# States: 0=normal, 1=high, 2=failure
def get_state(cpu):
    if cpu < 50:
        return 0
    elif cpu < 75:
        return 1
    else:
        return 2

states = ["normal", "high", "failure"]

# Actions: 0=do nothing, 1=restart, 2=scale
actions = ["do_nothing", "restart", "scale"]

# Q-table
Q = np.zeros((3, 3))

# Reward function
def get_reward(state, action, anomaly):
    if state == 2 and action == 1:
        return 10
    elif state == 1 and action == 2:
        return 5
    elif state == 0 and action == 0:
        return 2
    elif anomaly:
        return -10
    else:
        return -5

# RL parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.01
if epsilon > epsilon_min:
    epsilon *= epsilon_decay

# =========================================================
# 5. TRAIN RL AGENT
# =========================================================
rewards_history = []

print("\nTraining RL agent...\n")

for episode in range(150):
    total_reward = 0

    for cpu in cpu_data:
        state = get_state(cpu)
        anomaly = is_anomaly(cpu)

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Q[state])

        reward = get_reward(state, action, anomaly)
        total_reward += reward

        next_state = get_state(cpu)

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

    rewards_history.append(total_reward)

# =========================================================
# 6. SHOW LEARNED POLICY
# =========================================================
print("\n=== Learned Policy ===\n")

for i, s in enumerate(states):
    best_action = actions[np.argmax(Q[i])]
    print(f"{s.upper()} → {best_action}")

# =========================================================
# 7. SYSTEM DECISION SIMULATION
# =========================================================
print("\n=== Sample System Decisions ===\n")

for cpu in cpu_data[:10]:
    state = get_state(cpu)
    action = np.argmax(Q[state])

    print(f"CPU: {round(cpu,2)} | State: {states[state]} | Action: {actions[action]}")

# =========================================================
# 8. VISUALIZATION
# =========================================================

# RL learning curve
plt.figure()
plt.plot(rewards_history)
plt.title("RL Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")

# CPU vs Prediction
plt.figure()
plt.plot(cpu_data[10:], label="Actual CPU")
plt.plot(predicted.flatten(), label="Predicted CPU")
plt.legend()
plt.title("CPU Prediction (LSTM)")

plt.show()
