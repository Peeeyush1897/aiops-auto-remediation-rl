# =========================================================
# AIOps-Based Intelligent Auto-Remediation System
# RL-Centric Implementation (Thesis Ready)
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================================
# 1. DATA COLLECTION (SIMULATION)
# =========================================================
np.random.seed(42)

cpu = np.random.normal(60, 15, 300)
memory = np.random.normal(65, 10, 300)

# =========================================================
# 2. PREDICTION LAYER (LSTM)
# =========================================================
scaler = MinMaxScaler()
cpu_scaled = scaler.fit_transform(cpu.reshape(-1, 1))

X, y = [], []
window = 10

for i in range(window, len(cpu_scaled)):
    X.append(cpu_scaled[i-window:i])
    y.append(cpu_scaled[i])

X, y = np.array(X), np.array(y)

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X.shape[1], 1)))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mse')

print("Training LSTM...")
lstm.fit(X, y, epochs=5, verbose=0)

predicted = lstm.predict(X)
predicted = scaler.inverse_transform(predicted)

# =========================================================
# 3. ANOMALY DETECTION (DBSCAN)
# =========================================================
data_points = np.column_stack((cpu, memory))

dbscan = DBSCAN(eps=10, min_samples=5)
labels = dbscan.fit_predict(data_points)

# -1 means anomaly
anomalies = (labels == -1)

# =========================================================
# 4. ROOT CAUSE ANALYSIS (SIMPLIFIED)
# =========================================================
def get_state(cpu_value):
    if cpu_value < 50:
        return "normal"
    elif cpu_value < 80:
        return "high_cpu"
    else:
        return "failure"

# =========================================================
# 5. REINFORCEMENT LEARNING (Q-LEARNING)
# =========================================================
states = ["normal", "high_cpu", "failure"]
actions = ["do_nothing", "restart", "scale"]

Q = np.zeros((len(states), len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_reward(state, action):
    if state == "failure":
        return 10 if action in ["restart", "scale"] else -10
    elif state == "high_cpu":
        return 5 if action == "scale" else -5
    else:
        return 2 if action == "do_nothing" else -2

# Training RL
episodes = 100
rl_rewards = []

for ep in range(episodes):
    total_reward = 0
    
    for i in range(len(cpu)):
        state = get_state(cpu[i])
        s = states.index(state)
        
        # epsilon-greedy
        if random.uniform(0,1) < epsilon:
            a = random.randint(0, len(actions)-1)
        else:
            a = np.argmax(Q[s])
        
        reward = get_reward(state, actions[a])
        total_reward += reward
        
        # next state
        next_state = get_state(cpu[i])
        ns = states.index(next_state)
        
        # Q-learning update
        Q[s,a] = Q[s,a] + alpha * (reward + gamma * np.max(Q[ns]) - Q[s,a])
    
    rl_rewards.append(total_reward)

# =========================================================
# 6. AUTO-REMEDIATION (SIMULATION)
# =========================================================
def take_action(action):
    if action == "restart":
        return "Restarting container..."
    elif action == "scale":
        return "Scaling resources..."
    else:
        return "No action needed"

# Demo run
print("\n--- Sample Decisions ---")
for i in range(5):
    state = get_state(cpu[i])
    s = states.index(state)
    action = actions[np.argmax(Q[s])]
    print(f"State: {state} → Action: {action} → {take_action(action)}")

# =========================================================
# 7. VISUALIZATION
# =========================================================

# LSTM Prediction
plt.figure()
plt.plot(cpu[window:], label="Actual CPU")
plt.plot(predicted, label="Predicted CPU")
plt.title("LSTM Prediction")
plt.legend()
plt.show()

# Anomaly Detection
plt.figure()
plt.scatter(cpu, memory, c=labels)
plt.title("DBSCAN Anomaly Detection")
plt.show()

# RL Learning Curve
plt.figure()
plt.plot(rl_rewards)
plt.title("RL Learning Performance")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
