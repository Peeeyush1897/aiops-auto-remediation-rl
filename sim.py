"""
aiops_system.py
---------------
AIOps-based intelligent auto-remediation prototype.

Pipeline:
    simulate data  →  LSTM prediction  →  DBSCAN anomaly detection
    →  Q-learning remediation policy  →  action dispatch

Author note:  this is a research prototype, not production code.
              docker/k8s hooks are stubbed out in take_action().
"""

import logging
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

#import config as cfg
# ── config (inlined) ──────────────────────────────────────────────────
class cfg:
    SEED        = 42
    N_SAMPLES   = 300
    WINDOW_SIZE = 10
    LSTM_UNITS  = 32
    EPOCHS      = 5
    DBSCAN_EPS  = 10
    DBSCAN_MINS = 5
    ALPHA       = 0.1
    GAMMA       = 0.9
    EPSILON     = 0.2
    EPISODES    = 100
    CPU_HIGH    = 50
    CPU_FAIL    = 80
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

os.makedirs("outputs", exist_ok=True)


# =========================================================================== #
#  1. DATA SIMULATION                                                          #
# =========================================================================== #

def simulate_metrics(n=cfg.N_SAMPLES):
    """
    Generate synthetic CPU / memory readings.
    Real deployment would pull from Prometheus or Datadog instead.
    """
    cpu    = np.random.normal(60, 15, n).clip(0, 100)
    memory = np.random.normal(65, 10, n).clip(0, 100)
    log.info("Simulated %d data points  (cpu mean=%.1f, mem mean=%.1f)",
             n, cpu.mean(), memory.mean())
    return cpu, memory


# =========================================================================== #
#  2. LSTM PREDICTION                                                          #
# =========================================================================== #

def build_sequences(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm(cpu_raw):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cpu_raw.reshape(-1, 1))

    X, y = build_sequences(scaled, cfg.WINDOW_SIZE)
    # LSTM expects (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(cfg.LSTM_UNITS, input_shape=(X.shape[1], 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    log.info("Training LSTM  (epochs=%d) …", cfg.EPOCHS)
    model.fit(X, y, epochs=cfg.EPOCHS, verbose=0)

    predicted_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled)
    log.info("LSTM training done.  Prediction shape: %s", predicted.shape)
    return predicted, model, scaler


# =========================================================================== #
#  3. ANOMALY DETECTION (DBSCAN)                                               #
# =========================================================================== #

def detect_anomalies(cpu, memory):
    """
    DBSCAN clusters on (cpu, memory) pairs.
    Points labelled -1 are outliers → anomalies.
    """
    points = np.column_stack((cpu, memory))
    db = DBSCAN(eps=cfg.DBSCAN_EPS, min_samples=cfg.DBSCAN_MINS)
    labels = db.fit_predict(points)

    n_anomalies = (labels == -1).sum()
    log.info("DBSCAN found %d anomalies out of %d points", n_anomalies, len(labels))
    return labels


# =========================================================================== #
#  4. ROOT CAUSE CLASSIFICATION                                                #
# =========================================================================== #

# Keep thresholds in config so ops team can tune without reading the code
STATES  = ["normal", "high_cpu", "failure"]
ACTIONS = ["do_nothing", "restart", "scale"]


def classify_state(cpu_val: float) -> str:
    if cpu_val >= cfg.CPU_FAIL:
        return "failure"
    if cpu_val >= cfg.CPU_HIGH:
        return "high_cpu"
    return "normal"


# =========================================================================== #
#  5. Q-LEARNING AGENT                                                         #
# =========================================================================== #

def reward_fn(state: str, action: str) -> float:
    """
    Hand-tuned reward table.  Positive rewards steer the agent toward
    the correct remediation; negative rewards discourage wrong actions.
    """
    rules = {
        "failure":  {"restart": 10, "scale": 10, "do_nothing": -10},
        "high_cpu": {"scale": 5,    "restart": -5, "do_nothing": -5},
        "normal":   {"do_nothing": 2, "restart": -2, "scale": -2},
    }
    return rules[state][action]


def train_qlearning(cpu: np.ndarray):
    n_states  = len(STATES)
    n_actions = len(ACTIONS)
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for ep in range(cfg.EPISODES):
        total = 0.0
        for val in cpu:
            s = STATES.index(classify_state(val))

            # ε-greedy action selection
            if random.random() < cfg.EPSILON:
                a = random.randrange(n_actions)
            else:
                a = int(np.argmax(Q[s]))

            r  = reward_fn(STATES[s], ACTIONS[a])
            total += r

            # next state: same observation (tabular, single-step episodes)
            ns = s
            Q[s, a] += cfg.ALPHA * (r + cfg.GAMMA * np.max(Q[ns]) - Q[s, a])

        rewards_per_episode.append(total)

    log.info("Q-learning done.  Final Q-table:\n%s", Q.round(2))
    return Q, rewards_per_episode


# =========================================================================== #
#  6. AUTO-REMEDIATION DISPATCH                                                #
# =========================================================================== #

# TODO: wire restart/scale to actual kubectl / docker SDK calls
def take_action(action: str) -> str:
    dispatch = {
        "restart":    "Restarting container  [stub: docker restart <id>]",
        "scale":      "Scaling out           [stub: kubectl scale --replicas=+1]",
        "do_nothing": "System nominal – no action taken",
    }
    return dispatch.get(action, "Unknown action")


def run_remediation_demo(cpu: np.ndarray, Q: np.ndarray, n_samples: int = 10):
    print("\n" + "─" * 52)
    print(f"{'STATE':<12}  {'ACTION':<12}  OUTCOME")
    print("─" * 52)
    for val in cpu[:n_samples]:
        state  = classify_state(val)
        s      = STATES.index(state)
        action = ACTIONS[int(np.argmax(Q[s]))]
        print(f"{state:<12}  {action:<12}  {take_action(action)}")
    print("─" * 52)


# =========================================================================== #
#  7. VISUALISATION                                                            #
# =========================================================================== #

def plot_lstm(cpu, predicted, window):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(cpu[window:], label="Actual CPU",    linewidth=1.2, alpha=0.8)
    ax.plot(predicted,    label="Predicted CPU", linewidth=1.5, linestyle="--")
    ax.set_title("LSTM One-Step-Ahead CPU Prediction")
    ax.set_xlabel("Time step")
    ax.set_ylabel("CPU utilisation (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = "outputs/lstm_prediction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_anomalies(cpu, memory, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(cpu, memory, c=labels, cmap="coolwarm", s=18, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label="Cluster label  (-1 = anomaly)")
    ax.set_title("DBSCAN Anomaly Detection (CPU vs Memory)")
    ax.set_xlabel("CPU utilisation (%)")
    ax.set_ylabel("Memory utilisation (%)")
    fig.tight_layout()
    path = "outputs/anomaly_detection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_rl_curve(rewards):
    window = max(1, len(rewards) // 10)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards,  alpha=0.35, color="steelblue", label="Episode reward")
    ax.plot(range(window - 1, len(rewards)), smoothed,
            color="steelblue", linewidth=2, label=f"Moving avg ({window} ep)")
    ax.set_title("Q-Learning Convergence Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = "outputs/rl_learning_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", path)


# =========================================================================== #
#  ENTRY POINT                                                                 #
# =========================================================================== #

if __name__ == "__main__":
    # 1. Data
    cpu, memory = simulate_metrics()

    # 2. LSTM prediction
    predicted, lstm_model, scaler = train_lstm(cpu)
    plot_lstm(cpu, predicted, cfg.WINDOW_SIZE)

    # 3. Anomaly detection
    labels = detect_anomalies(cpu, memory)
    plot_anomalies(cpu, memory, labels)

    # 4. Q-learning
    Q, rl_rewards = train_qlearning(cpu)
    plot_rl_curve(rl_rewards)

    # 5. Demo decisions
    run_remediation_demo(cpu, Q, n_samples=10)

    log.info("All outputs written to outputs/")
    sys.exit(0)
