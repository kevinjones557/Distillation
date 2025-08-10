import re
import matplotlib.pyplot as plt
import numpy as np

# Parameters
log_path = '../logs/train_new.out'
num_average_train = 1000  # <-- Adjust this for smoothing window size
num_average_val = 1

# Storage for values
train_steps = []
train_losses = []

val_steps = []
val_losses = []

current_tokens = None
alpha = 0

# Regex
train_re = re.compile(r"Hard loss: ([\d.]+),\s+Soft loss: ([\d.]+).*Tokens Processed \(total\): ([\d.]+) M")
val_re = re.compile(r"\[Validation Loss\] Hard: ([\d.]+), Soft: ([\d.]+)")

with open(log_path, 'r') as f:
    for line in f:
        train_match = train_re.search(line)
        val_match = val_re.search(line)

        if train_match:
            hard, soft, tokens = map(float, train_match.groups())
            step = tokens
            train_steps.append(step)
            train_losses.append(soft * alpha + hard * (1-alpha))
            current_tokens = step
        elif val_match and current_tokens is not None:
            val_hard, val_soft = map(float, val_match.groups())
            val_steps.append(current_tokens)
            val_losses.append(val_soft * alpha + val_hard * (1-alpha))

# Smoothing with moving average
def moving_average(values, window):
    return np.convolve(values, np.ones(window)/window, mode='valid')

smoothed_steps = train_steps[num_average_train-1:]  # Align steps with averaged losses
train_loss_avgs = moving_average(train_losses, num_average_train)
val_loss_avgs = moving_average(val_losses, num_average_val)
smoothed_val_steps = val_steps[num_average_val - 1:]
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(smoothed_steps, train_loss_avgs, label='Training Hard Loss (avg)', color='blue')
plt.plot(smoothed_val_steps, val_loss_avgs, label='Validation Hard Loss', color='red')

plt.xlabel("Tokens Processed (Millions)")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/hard_only.png")
