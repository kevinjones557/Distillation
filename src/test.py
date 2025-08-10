import os

total_tokens = 1000
iteration = 700

switched_to_hard_loss_only_file = '../switched.info'
if not os.path.exists(switched_to_hard_loss_only_file):
    with open(switched_to_hard_loss_only_file, "w") as file:
        file.write(f"Started hard Loss only at iteration {iteration} and token count {total_tokens}")
        file.flush()
        file.close()