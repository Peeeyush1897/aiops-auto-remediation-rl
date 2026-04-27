# ============================================
# Optimized AIOps RL System (Thesis Version)
# ============================================

import numpy as np
import random
import matplotlib.pyplot as plt

# ---------------------------
# 1. Simulated System Metrics
# ---------------------------
np.random.seed(42)
cpu_data = np.random.normal(60, 15, 200)  # CPU usage

# ---------------------------
# 2. State Definition
# ---------------------------
def get_state(cpu):
    if cpu < 50:
        return 0  # normal
    elif cpu < 75:
        return 1  # high load
    else:
        return 2  # failure

states = ["normal", "high", "failure"]

# ---------------------------
# 3. Actions
# ---------------------------
actions = ["do_nothing", "restart", "scale"]

# ---------------------------
# 4. Q-table (RL Brain)
# ---------------------------
Q = np.zeros((3, 3))

# ---------------------------
# 5. Anomaly Detection (Simple)
# ---------------------------
def is_anomaly(cpu):
    return cpu > 80  # threshold

# ---------------------------
# 6. Reward Function
# ---------------------------
def reward(state, action, anomaly):
    
    if state == 2 and action == 1:  # failure → restart
        return 10
    elif state == 1 and action == 2:  # high → scale
        return 5
    elif state == 0 and action == 0:  # normal → do nothing
        return 2
    elif anomaly:
        return -10  # anomaly penalty
    else:
        return -5

# ---------------------------
# 7. RL Parameters
# ---------------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# ---------------------------
# 8. Training Loop
# ---------------------------
rewards_history = []

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

        r = reward(state, action, anomaly)
        total_reward += r

        next_state = get_state(cpu)

        # Q-learning update
        Q[state, action] += alpha * (
            r + gamma * np.max(Q[next_state]) - Q[state, action]
        )

    rewards_history.append(total_reward)

# ---------------------------
# 9. Learned Policy
# ---------------------------
print("\n=== Learned Policy ===\n")

for i, s in enumerate(states):
    best_action = actions[np.argmax(Q[i])]
    print(f"{s.upper()} → {best_action}")

# ---------------------------
# 10. Simulation (Decisions)
# ---------------------------
print("\n=== Sample Decisions ===\n")

for cpu in cpu_data[:10]:
    state = get_state(cpu)
    action = np.argmax(Q[state])
    
    print(f"CPU: {round(cpu,2)} | State: {states[state]} | Action: {actions[action]}")

# ---------------------------
# 11. Visualization
# ---------------------------
plt.plot(rewards_history)
plt.title("RL Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
