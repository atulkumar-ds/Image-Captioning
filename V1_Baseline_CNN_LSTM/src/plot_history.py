import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent

DEFAULT_HISTORY_FILE = V1_ROOT / "models" / "history.json"
DEFAULT_PLOTS_DIR = V1_ROOT / "plots"


def plot_training_history(history_file=None, output_dir=None):
    if history_file is None:
        history_file = DEFAULT_HISTORY_FILE
    if output_dir is None:
        output_dir = DEFAULT_PLOTS_DIR

    history_file = Path(history_file).resolve()
    output_dir = Path(output_dir).resolve()

    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading training history from: {history_file}")
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)

    # 1. Plot Loss
    if "loss" in history and "val_loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["loss"], marker="o", linewidth=2, label="Training Loss")
        plt.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        loss_path = output_dir / "loss.png"
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved to: {loss_path}")

    # 2. Plot Accuracy (if available)
    if "accuracy" in history and "val_accuracy" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["accuracy"], marker="o", linewidth=2, label="Training Accuracy")
        plt.plot(epochs, history["val_accuracy"], marker="o", linewidth=2, label="Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        accuracy_path = output_dir / "accuracy.png"
        plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Accuracy plot saved to: {accuracy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history curves.")
    parser.add_argument("--history", type=str, default=str(DEFAULT_HISTORY_FILE), help="Path to history.json")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PLOTS_DIR), help="Output directory for plots")
    args = parser.parse_args()

    plot_training_history(args.history, args.output_dir)
