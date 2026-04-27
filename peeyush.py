import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# ==============================
# 1. DQN AGENT
# ==============================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            states.append(state[0])
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==============================
# 2. ENVIRONMENT
# ==============================
class DevOpsEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.array([0.5, 0.5])  # cpu, memory
        return self.state

    def step(self, action):
        cpu, mem = self.state

        cpu += np.random.normal(0, 0.1)
        mem += np.random.normal(0, 0.1)

        cpu = np.clip(cpu, 0, 1)
        mem = np.clip(mem, 0, 1)

        # reward logic
        if cpu > 0.8 and action == 1:
            reward = 10
        elif cpu > 0.6 and action == 2:
            reward = 5
        elif cpu < 0.6 and action == 0:
            reward = 2
        else:
            reward = -5

        self.state = np.array([cpu, mem])
        done = False

        return self.state, reward, done

# ==============================
# 3. LSTM MODEL (ANOMALY)
# ==============================
def train_lstm(data):
    X, y = [], []

    for i in range(10, len(data)):
        X.append(data[i-10:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=5, verbose=0)

    return model

def detect_anomaly(value):
    return value > 0.8

# ==============================
# 4. TRAINING LOOP
# ==============================
state_size = 2
action_size = 3  # do nothing, restart, scale

agent = DQNAgent(state_size, action_size)
env = DevOpsEnv()

episodes = 50
rewards = []

# generate fake CPU history for LSTM
cpu_history = np.random.rand(200, 1)
lstm_model = train_lstm(cpu_history)

for e in range(episodes):
    state = env.reset().reshape(1, -1)
    total_reward = 0

    for time in range(50):
        action = agent.act(state)

        next_state, reward, done = env.step(action)
        next_state = next_state.reshape(1, -1)

        # LSTM anomaly prediction
        pred = lstm_model.predict(np.array([cpu_history[-10:]]), verbose=0)[0][0]

        if detect_anomaly(pred):
            reward += 3

        agent.remember(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    agent.replay()
    rewards.append(total_reward)

    print(f"Episode {e+1}, Reward: {total_reward}")

# ==============================
# 5. PLOT
# ==============================
plt.plot(rewards)
plt.title("DQN + LSTM Auto-Remediation")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
