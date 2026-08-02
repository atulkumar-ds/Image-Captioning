import os
import json
import matplotlib.pyplot as plt


HISTORY_FILE = "models/history.json"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


print("Loading training history...")

with open(
    HISTORY_FILE,
    "r",
    encoding="utf-8"
) as file:

    history = json.load(file)


accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(
    1,
    len(accuracy) + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    accuracy,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

accuracy_path = os.path.join(
    PLOTS_DIR,
    "accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

loss_path = os.path.join(
    PLOTS_DIR,
    "loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nPlots generated successfully!")

print("\nSaved files:")
print(accuracy_path)
print(loss_path)