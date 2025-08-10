import re
import matplotlib.pyplot as plt
import os

# Replace with your actual log file path
# file_names = ["no_teacher_train.out", "teacher_train_alpha(.5)_temp(1).out", \
#               "teacher_train_alpha(.8)_temp(1).out", "teacher_train_alpha(.8)_temp(.5).out", \
#               "teacher_train_alpha(.8)_temp(2).out", "full_model_loss.out"]

file_names = ["train_new.out"]
# file_names = [f for f in os.listdir("logs") if 'kldiv' in f]

# Pattern to match "Hard loss: x.xx"
hard_loss_pattern = re.compile(r"Hard loss:\s*([0-9.]+)")
soft_loss_pattern = re.compile(r"Soft loss:\s*([0-9.]+)")

for log_file in file_names:
    hard_losses = []

    # Read and parse the log file
    with open(f"../logs/{log_file}", "r") as f:
        i = 0
        temp_losses = []
        for line in f:
            soft_match = soft_loss_pattern.search(line)
            hard_match = hard_loss_pattern.search(line)
            if soft_match and hard_match:
                i += 1
                if i % 100 == 0:
                    i = 0
                    hard_losses.append(sum(temp_losses) / len(temp_losses))
                    temp_losses = []
                temp_losses.append(float(hard_match.group(1)))# * 0.8 + float(hard_match.group(1)) * 0.2)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.ylim(2, 12)
    plt.plot(hard_losses, label="Hard Loss", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Hard Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../images/{log_file[:-4].replace('.', '%')}.png")
