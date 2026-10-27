import os
import sys
import json
import random
import hashlib
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent
REPO_ROOT = V1_ROOT.parent

# Set working directory to V1_ROOT so legacy dataset.py relative paths resolve
os.chdir(V1_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from V1_Baseline_CNN_LSTM.src.dataset import load_captions, clean_all_captions
    from V1_Baseline_CNN_LSTM.src.vocabulary import build_vocabulary
    from V1_Baseline_CNN_LSTM.src.tokenizer import create_tokenizer
    from V1_Baseline_CNN_LSTM.src.sequences import find_max_length
    from V1_Baseline_CNN_LSTM.src.cnn_features import load_features
    from V1_Baseline_CNN_LSTM.src.data_generator import CaptionDataGenerator
    from V1_Baseline_CNN_LSTM.src.model import build_caption_model, compile_caption_model
    from V1_Baseline_CNN_LSTM.src.save_tokenizer import save_tokenizer
    from V1_Baseline_CNN_LSTM.src.bleu_score import compute_split_manifest_fingerprint, save_experiment_config_safely
except ImportError:
    from dataset import load_captions, clean_all_captions
    from vocabulary import build_vocabulary
    from tokenizer import create_tokenizer
    from sequences import find_max_length
    from cnn_features import load_features
    from data_generator import CaptionDataGenerator
    from model import build_caption_model, compile_caption_model
    from save_tokenizer import save_tokenizer
    from bleu_score import compute_split_manifest_fingerprint, save_experiment_config_safely

DEFAULT_SPLITS_FILE = REPO_ROOT / "splits.json"
DEFAULT_FEATURE_FILE = V1_ROOT / "data" / "features" / "image_features.npz"
HISTORICAL_MODELS_DIR = (V1_ROOT / "models").resolve()


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU detected:", gpus[0])
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected or available on this platform — using CPU")


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Train V1 Baseline CNN-LSTM Captioning Model.")
    parser.add_argument("--split", type=str, default="canonical", choices=["canonical", "historical"],
                        help="Dataset split to train on (default: canonical)")
    parser.add_argument("--splits_file", type=str, default=str(DEFAULT_SPLITS_FILE),
                        help="Path to splits.json manifest")
    parser.add_argument("--output_dir", type=str, default=str(V1_ROOT / "experiments" / "v1_canonical"),
                        help="Output directory for new experiment artifacts")
    parser.add_argument("--features_file", type=str, default=str(DEFAULT_FEATURE_FILE),
                        help="Path to precomputed features .npz (default: data/features/image_features.npz)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--min_frequency", type=int, default=5, help="Minimum token frequency for vocabulary")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Word embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden state dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for algorithmic reproducibility (default: 42)")
    parser.add_argument("--dry_run", action="store_true", help="Audit setup and build model without training")
    parser.add_argument("--allow_overwrite_historical", action="store_true",
                        help="Explicit flag required to overwrite historical models/ directory")
    parser.add_argument("--allow_overwrite_experiment", action="store_true",
                        help="Explicit flag required to overwrite an existing completed experiment directory")
    return parser.parse_args(args)


def load_dataset_split(split_type, splits_file_path, captions_dict):
    splits_file = Path(splits_file_path).resolve()
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits manifest not found at {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    if split_type == "canonical":
        split_info = splits_data.get("canonical_split", {})
        train_images = split_info.get("train", [])
        val_images = split_info.get("val", [])
        test_images = split_info.get("test", [])

        # Integrity checks
        assert set(train_images).isdisjoint(set(val_images)), "Canonical train and val overlap!"
        assert set(train_images).isdisjoint(set(test_images)), "Canonical train and test overlap!"
        assert set(val_images).isdisjoint(set(test_images)), "Canonical val and test overlap!"
        assert len(train_images) == 5663, f"Expected 5663 train images, got {len(train_images)}"
        assert len(val_images) == 1213, f"Expected 1213 val images, got {len(val_images)}"
        assert len(test_images) == 1215, f"Expected 1215 test images, got {len(test_images)}"

        print("\nCanonical Split Loaded Successfully:")
        print(f"  Training Images   : {len(train_images)} (70%)")
        print(f"  Validation Images : {len(val_images)} (15%)")
        print(f"  Test Images (HELD): {len(test_images)} (15% - Strictly excluded from training/val)")

    else:
        split_info = splits_data.get("historical_v1_split", {})
        train_images = split_info.get("train", [])
        val_images = split_info.get("val", [])
        test_images = []
        assert set(train_images).isdisjoint(set(val_images)), "Historical train and val overlap!"
        print("\nHistorical V1 Split Loaded:")
        print(f"  Training Images   : {len(train_images)} (90%)")
        print(f"  Validation Images : {len(val_images)} (10%)")

    train_captions = {img: captions_dict[img] for img in train_images if img in captions_dict}
    val_captions = {img: captions_dict[img] for img in val_images if img in captions_dict}

    return train_captions, val_captions, split_info


def validate_training_setup(args):
    """
    Non-mutating validation-only dry-run verification.
    Validates hyperparameters, split manifests, caption paths, vocabulary construction,
    and feature paths without building models, initializing GPU, or writing any files to disk.
    """
    if args.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")
    if args.min_frequency < 1:
        raise ValueError(f"min_frequency must be >= 1, got {args.min_frequency}")
    if args.embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be > 0, got {args.embedding_dim}")
    if args.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {args.hidden_dim}")
    seed = getattr(args, "seed", 42)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")

    splits_file = Path(args.splits_file).resolve()
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits manifest not found at {splits_file}")

    print("\n[DRY RUN] Validating caption dataset...")
    captions_dict = clean_all_captions(load_captions())
    print(f"[DRY RUN] Total images with captions: {len(captions_dict)}")

    print(f"[DRY RUN] Validating split manifest '{splits_file.name}' for split '{args.split}'...")
    train_captions, val_captions, split_info = load_dataset_split(
        split_type=args.split,
        splits_file_path=args.splits_file,
        captions_dict=captions_dict
    )
    print(f"[DRY RUN] Training images: {len(train_captions)}, Validation images: {len(val_captions)}")

    print("\n[DRY RUN] Validating vocabulary construction (in-memory)...")
    word_counts = build_vocabulary(train_captions)
    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=args.min_frequency
    )
    vocab_size = len(word_to_index)
    max_length = find_max_length(train_captions)
    print(f"[DRY RUN] Vocabulary size: {vocab_size}, Maximum caption length: {max_length}")

    # Check feature file existence
    if hasattr(args, "features_file") and args.features_file:
        feat_path = Path(args.features_file).resolve()
        feat_exists = feat_path.exists()
        print(f"[DRY RUN] Feature file specified: {feat_path} (exists: {feat_exists})")
    else:
        print("[DRY RUN] Feature file: None specified")

    output_dir = Path(args.output_dir).resolve()
    print(f"[DRY RUN] Target output directory: {output_dir}")
    if output_dir.exists():
        existing_artifacts = [p.name for p in output_dir.glob("**/*") if p.is_file()]
        if existing_artifacts:
            print(f"[DRY RUN] Note: Target directory exists with {len(existing_artifacts)} existing file(s).")
            print("[DRY RUN] Existing artifacts left completely untouched by dry-run.")

    print("\n========================================================")
    print("[DRY RUN COMPLETE] Validation successful. Zero files written.")
    print("========================================================\n")
    return {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "train_images": len(train_captions),
        "val_images": len(val_captions)
    }


def set_reproducible_seed(seed=42):
    """
    Sets random seeds across Python random, NumPy, and TensorFlow.
    Controls algorithmic randomness (layer initialization, sequence shuffling, dropout masks).
    Note: Exact bit-for-bit numerical reproducibility across heterogeneous hardware,
    CUDA drivers, or multithreaded CPU kernels is subject to floating-point nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)


def compute_file_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_history_safely(history_data, target_path):
    """
    Atomically writes training history JSON to prevent partial or interrupted
    writes from corrupting completed epoch records.
    """
    target_path = Path(target_path).resolve()
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """
    Atomically writes training history and updates experiment config fingerprint
    after each completed epoch. Ensures that an interrupted or killed run leaves
    a valid, uncorrupted history record and valid checkpoint fingerprint up to
    the last completed epoch.
    """
    def __init__(self, output_dir, config_data, checkpoint_path):
        super().__init__()
        self.output_dir = Path(output_dir).resolve()
        self.config_data = config_data
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.history_records = {}
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            if k not in self.history_records:
                self.history_records[k] = []
            self.history_records[k].append(float(v))

        # Atomic history write after every epoch
        history_path = self.output_dir / "history.json"
        save_history_safely(self.history_records, history_path)

        # Update checkpoint fingerprint if a new best checkpoint was saved
        val_loss = logs.get("val_loss")
        if val_loss is not None and float(val_loss) < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            if self.checkpoint_path.exists():
                ckpt_h = compute_file_sha256(self.checkpoint_path)
                self.config_data["checkpoint_fingerprint"] = ckpt_h
                self.config_data["completed_epochs"] = epoch + 1
                self.config_data["best_val_loss"] = float(val_loss)
                save_experiment_config_safely(self.config_data, self.output_dir / "config.json")


def train_v1():
    args = parse_arguments()
    if args.dry_run:
        validate_training_setup(args)
        return

    output_dir = Path(args.output_dir).resolve()
    output_models_dir = output_dir / "models"
    output_logs_dir = output_dir / "logs"

    # Historical experiment protection guard
    if output_models_dir == HISTORICAL_MODELS_DIR and not args.allow_overwrite_historical:
        raise PermissionError(
            f"PROTECTION GUARD: Cannot overwrite historical models directory '{HISTORICAL_MODELS_DIR}'. "
            f"Specify a separate experiment output directory (e.g., --output_dir experiments/v1_canonical)."
        )

    # Experiment overwrite protection guard
    existing_config = output_dir / "config.json"
    if existing_config.exists() and not args.allow_overwrite_experiment and not args.allow_overwrite_historical:
        try:
            with open(existing_config, "r", encoding="utf-8") as f:
                cfg_prev = json.load(f)
            if cfg_prev.get("checkpoint_fingerprint"):
                raise FileExistsError(
                    f"EXPERIMENT OVERWRITE GUARD: Directory '{output_dir}' already contains a completed "
                    f"experiment (checkpoint_fingerprint: {cfg_prev['checkpoint_fingerprint'][:12]}...). "
                    f"To prevent accidental loss of experiment records, specify a different --output_dir "
                    f"or pass --allow_overwrite_experiment."
                )
        except (json.JSONDecodeError, KeyError):
            pass

    configure_gpu()
    print(f"\nSetting reproducible seed: {args.seed}...")
    set_reproducible_seed(args.seed)

    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs(output_logs_dir, exist_ok=True)

    print("\nLoading and cleaning captions...")
    captions_dict = clean_all_captions(load_captions())
    print("Total images with captions:", len(captions_dict))

    train_captions, val_captions, split_info = load_dataset_split(
        split_type=args.split,
        splits_file_path=args.splits_file,
        captions_dict=captions_dict
    )

    print("\nBuilding vocabulary (strictly from training captions)...")
    word_counts = build_vocabulary(train_captions)
    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=args.min_frequency
    )
    vocab_size = len(word_to_index)
    max_length = find_max_length(train_captions)

    print("Vocabulary size (including special tokens):", vocab_size)
    print("Maximum caption length:", max_length)

    tokenizer_save_path = output_models_dir / "tokenizer.json"
    save_tokenizer(
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        max_length=max_length,
        save_path=str(tokenizer_save_path)
    )

    print("\nBuilding model...")
    model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
        feature_size=2048,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    )
    model = compile_caption_model(model)
    model.summary()

    import hashlib

    # Deterministic fingerprints of split partitions
    train_split_fp = hashlib.sha256(json.dumps(sorted(train_captions.keys())).encode("utf-8")).hexdigest()
    val_split_fp = hashlib.sha256(json.dumps(sorted(val_captions.keys())).encode("utf-8")).hexdigest()
    test_imgs = split_info.get("test", [])
    test_split_fp = hashlib.sha256(json.dumps(sorted(test_imgs)).encode("utf-8")).hexdigest() if test_imgs else None
    split_manifest_fp = compute_split_manifest_fingerprint(split_info)

    # Tokenizer SHA-256 fingerprint
    tok_h = hashlib.sha256()
    with open(tokenizer_save_path, "rb") as f:
        while chunk := f.read(65536):
            tok_h.update(chunk)
    tok_fp = tok_h.hexdigest()

    config_data = {
        "model_version": "V1_Baseline_CNN_LSTM",
        "split_type": args.split,
        "split_manifest": os.path.relpath(Path(args.splits_file).resolve(), output_dir) if Path(args.splits_file).exists() else str(args.splits_file),
        "train_split_fingerprint": train_split_fp,
        "val_split_fingerprint": val_split_fp,
        "test_split_fingerprint": test_split_fp,
        "split_manifest_fingerprint": split_manifest_fp,
        "train_images": len(train_captions),
        "val_images": len(val_captions),
        "test_images": len(test_imgs),
        "seed": args.seed,
        "checkpoint_file": "models/best_caption_model.keras",
        "checkpoint_fingerprint": None,
        "tokenizer_file": "models/tokenizer.json",
        "tokenizer_fingerprint": tok_fp,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "completed_epochs": 0,
        "best_val_loss": None,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "min_frequency": args.min_frequency,
        "architecture": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "min_frequency": args.min_frequency
        },
        "loss_objective": "sparse_categorical_crossentropy",
        "best_checkpoint_criterion": {
            "metric": "val_loss",
            "mode": "min"
        },
        "history_file": "history.json",
        "special_tokens": {
            "start_token": "<start>",
            "end_token": "<end>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        }
    }
    save_experiment_config_safely(config_data, output_dir / "config.json")

    features_path = Path(args.features_file).resolve()
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {features_path}. Provide valid --features_file path to run full training."
        )

    print("\nLoading precomputed CNN features...")
    features_dict = load_features(str(features_path))
    print("CNN features loaded:", len(features_dict))

    print("\nCreating data generators...")
    train_generator = CaptionDataGenerator(
        captions_dict=train_captions,
        features_dict=features_dict,
        word_to_index=word_to_index,
        max_length=max_length,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed
    )

    val_generator = CaptionDataGenerator(
        captions_dict=val_captions,
        features_dict=features_dict,
        word_to_index=word_to_index,
        max_length=max_length,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Training batches:", len(train_generator))
    print("Validation batches:", len(val_generator))

    checkpoint = ModelCheckpoint(
        filepath=str(output_models_dir / "best_caption_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )

    progress_callback = TrainingProgressCallback(
        output_dir=output_dir,
        config_data=config_data,
        checkpoint_path=output_models_dir / "best_caption_model.keras"
    )

    print("\n========================================")
    print("Starting Model Training")
    print("========================================\n")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr, progress_callback],
        verbose=1
    )

    save_history_safely(history.history, output_dir / "history.json")

    final_model_path = output_models_dir / "final_caption_model.keras"
    model.save(str(final_model_path))

    # Record saved best checkpoint fingerprint in experiment metadata
    best_ckpt_file = output_models_dir / "best_caption_model.keras"
    if best_ckpt_file.exists():
        config_data["checkpoint_fingerprint"] = compute_file_sha256(best_ckpt_file)
        config_data["completed_epochs"] = len(history.history.get("loss", []))
        if "val_loss" in history.history and history.history["val_loss"]:
            config_data["best_val_loss"] = float(min(history.history["val_loss"]))
        save_experiment_config_safely(config_data, output_dir / "config.json")

    print("\nTraining complete.")
    print("Best model:", output_models_dir / "best_caption_model.keras")
    print("Final model:", final_model_path)


if __name__ == "__main__":
    train_v1()
