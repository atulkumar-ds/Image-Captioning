import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
V2_ROOT = SRC_DIR.parent

DEFAULT_HISTORY_FILE = V2_ROOT / "experiments" / "v2_canonical" / "history.json"
DEFAULT_PLOTS_DIR = V2_ROOT / "plots"


def plot_training_history(history_file=None, output_dir=None):
    if history_file is None:
        if DEFAULT_HISTORY_FILE.exists():
            history_file = DEFAULT_HISTORY_FILE
        else:
            history_file = V2_ROOT / "models" / "history.json"
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

    # Support both "train_loss" / "val_loss" (V2) and "loss" / "val_loss" (standard Keras)
    train_loss = history.get("train_loss") or history.get("loss")
    val_loss = history.get("val_loss")

    if train_loss is None or val_loss is None:
        raise ValueError(
            f"History file must contain training and validation loss curves. Found keys: {list(history.keys())}"
        )

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Training Loss")
    plt.plot(epochs, val_loss, marker="o", linewidth=2, label="Validation Loss")
    plt.title("V2 Attention: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked Loss")
    plt.grid(True)
    plt.legend()
    loss_path = output_dir / "loss.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to: {loss_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot V2 training history curves.")
    parser.add_argument("--history", type=str, default=str(DEFAULT_HISTORY_FILE), help="Path to history.json")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PLOTS_DIR), help="Output directory for plots")
    args = parser.parse_args()

    plot_training_history(args.history, args.output_dir)
