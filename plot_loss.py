import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
# Pass a path as an argument, or set DEFAULT_LOG below.
DEFAULT_LOG = "checkpoints/20260320_144204/loss_log.csv"   # e.g. "checkpoints/20260319_000252/loss_log.csv"
SMOOTH_WINDOW = 10

# ── Load ──────────────────────────────────────────────────────────────────────
log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
if not log_path or not os.path.exists(log_path):
    print("Usage: python plot_loss.py <path/to/loss_log.csv>")
    sys.exit(1)

iterations, p_losses, v_losses, t_losses = [], [], [], []
with open(log_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iterations.append(int(row["iteration"]))
        p_losses.append(float(row["policy_loss"]))
        v_losses.append(float(row["value_loss"]))
        t_losses.append(float(row["total_loss"]))

# ── Smooth ────────────────────────────────────────────────────────────────────
def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

x       = iterations[SMOOTH_WINDOW - 1:]
p_smooth = smooth(p_losses, SMOOTH_WINDOW)
v_smooth = smooth(v_losses, SMOOTH_WINDOW)
t_smooth = smooth(t_losses, SMOOTH_WINDOW)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, p_smooth, label="Policy loss",  color="steelblue")
ax.plot(x, v_smooth, label="Value loss",   color="tomato")
ax.plot(x, t_smooth, label="Total loss",   color="mediumseagreen")

ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.legend()
ax.grid(alpha=0.3)

out_path = os.path.join(os.path.dirname(log_path), "loss_curve.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
