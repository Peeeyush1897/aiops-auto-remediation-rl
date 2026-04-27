"""
sim.py
------
AIOps-based intelligent auto-remediation prototype.

Pipeline:
    simulate data  ->  LSTM prediction  ->  DBSCAN anomaly detection
    ->  Q-learning remediation policy  ->  action dispatch

Author note: this is a research prototype, not production code.
             docker/k8s hooks are stubbed out in take_action().
"""

import logging
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Sequential

# --------------------------------------------------------------------------- #
#  CONFIG                                                                      #
# --------------------------------------------------------------------------- #

class cfg:
    SEED        = 42
    N_SAMPLES   = 300

    # LSTM
    WINDOW_SIZE = 10
    LSTM_UNITS  = 32
    EPOCHS      = 5

    # DBSCAN
    DBSCAN_EPS  = 10
    DBSCAN_MINS = 5

    # Q-Learning
    ALPHA       = 0.1
    GAMMA       = 0.9
    EPSILON     = 0.2
    EPISODES    = 100

    # State thresholds
    CPU_HIGH    = 50
    CPU_FAIL    = 80

# --------------------------------------------------------------------------- #
#  LOGGING                                                                     #
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

# --------------------------------------------------------------------------- #
#  STATES / ACTIONS                                                            #
# --------------------------------------------------------------------------- #

STATES  = ["normal", "high_cpu", "failure"]
ACTIONS = ["do_nothing", "restart", "scale"]

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
        Input(shape=(X.shape[1], 1)),
        LSTM(cfg.LSTM_UNITS),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    log.info("Training LSTM  (epochs=%d) ...", cfg.EPOCHS)
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
    Points labelled -1 are outliers -> anomalies.
    """
    points = np.column_stack((cpu, memory))
    db     = DBSCAN(eps=cfg.DBSCAN_EPS, min_samples=cfg.DBSCAN_MINS)
    labels = db.fit_predict(points)

    n_anomalies = (labels == -1).sum()
    log.info("DBSCAN found %d anomalies out of %d points", n_anomalies, len(labels))
    return labels

# =========================================================================== #
#  4. ROOT CAUSE CLASSIFICATION                                                #
# =========================================================================== #

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
    Hand-tuned reward table. Positive rewards steer the agent toward
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

            # epsilon-greedy action selection
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
        "do_nothing": "System nominal - no action taken",
    }
    return dispatch.get(action, "Unknown action")


def run_remediation_demo(cpu: np.ndarray, Q: np.ndarray, n_samples: int = 10):
    print("\n" + "-" * 52)
    print(f"{'STATE':<12}  {'ACTION':<12}  OUTCOME")
    print("-" * 52)
    for val in cpu[:n_samples]:
        state  = classify_state(val)
        s      = STATES.index(state)
        action = ACTIONS[int(np.argmax(Q[s]))]
        print(f"{state:<12}  {action:<12}  {take_action(action)}")
    print("-" * 52)

# =========================================================================== #
#  7. VISUALISATION                                                            #
# =========================================================================== #

def plot_lstm(cpu, predicted, window):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("LSTM One-Step-Ahead CPU Prediction", fontsize=13, fontweight="bold")

    actual = cpu[window:]
    errors = actual - predicted.flatten()

    # ── Top: actual vs predicted ─────────────────────────────────────
    ax = axes[0]
    ax.plot(actual,    label="Actual CPU",    linewidth=1.4, alpha=0.85, color="#378ADD")
    ax.plot(predicted, label="Predicted CPU", linewidth=1.5, linestyle="--", color="#D85A30")

    # Mark anomalies (CPU > 80)
    anom_idx = [i for i, v in enumerate(actual) if v > 80]
    if anom_idx:
        ax.scatter(anom_idx, actual[anom_idx],
                   color="#E24B4A", s=35, zorder=5, label="Anomaly (CPU > 80%)")

    ax.set_ylabel("CPU utilisation (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Bottom: prediction error bar chart ───────────────────────────
    ax2 = axes[1]
    colors = ["#378ADD" if e >= 0 else "#D85A30" for e in errors]
    ax2.bar(range(len(errors)), errors, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Error (%)")
    ax2.set_xlabel("Time step")
    ax2.grid(True, linestyle="--", alpha=0.3)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ax2.set_title(f"Prediction error (actual − predicted)   RMSE = {rmse:.2f}%",
                  fontsize=9, loc="left")

    fig.tight_layout()
    path = "outputs/lstm_prediction.png"
    fig.savefig(path, dpi=150)
    plt.show()                          # pops up on screen
    plt.close(fig)
    log.info("Saved -> %s", path)


def plot_anomalies(cpu, memory, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    normal  = labels != -1
    anomaly = labels == -1
    ax.scatter(cpu[normal],  memory[normal],  c="#1D9E75", s=18, alpha=0.6, label="Normal")
    ax.scatter(cpu[anomaly], memory[anomaly], c="#E24B4A", s=55, marker="^",
               alpha=0.9, label=f"Anomaly ({anomaly.sum()} pts)")
    ax.set_title("DBSCAN Anomaly Detection (CPU vs Memory)", fontsize=12)
    ax.set_xlabel("CPU utilisation (%)")
    ax.set_ylabel("Memory utilisation (%)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = "outputs/anomaly_detection.png"
    fig.savefig(path, dpi=150)
    plt.show()                          # pops up on screen
    plt.close(fig)
    log.info("Saved -> %s", path)


def plot_rl_curve(rewards):
    window   = max(1, len(rewards) // 10)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.35, color="steelblue", label="Episode reward")
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
    plt.show()                          # pops up on screen
    plt.close(fig)
    log.info("Saved -> %s", path)

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
