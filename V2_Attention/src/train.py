import os
import sys
import time
import math
import json
import random
import hashlib
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

SRC_DIR = Path(__file__).resolve().parent
V2_ROOT = SRC_DIR.parent
REPO_ROOT = V2_ROOT.parent

# Set working directory to V2_ROOT so legacy dataset.py relative paths resolve
os.chdir(V2_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from V2_Attention.src.dataset import load_captions, clean_all_captions
    from V2_Attention.src.vocabulary import build_vocabulary
    from V2_Attention.src.tokenizer import create_tokenizer
    from V2_Attention.src.sequences import find_max_length
    from V2_Attention.src.cnn_features import load_features
    from V2_Attention.src.training_data import build_training_dataset
    from V2_Attention.src.model import ImageCaptionModel
    from V2_Attention.src.train_step import train_step
    from V2_Attention.src.loss import loss_function, compute_loss_components, aggregate_validation_loss
    from V2_Attention.src.save_tokenizer import save_tokenizer
    from V2_Attention.src.evaluate import compute_split_manifest_fingerprint, save_experiment_config_safely
except ImportError:
    from dataset import load_captions, clean_all_captions
    from vocabulary import build_vocabulary
    from tokenizer import create_tokenizer
    from sequences import find_max_length
    from cnn_features import load_features
    from training_data import build_training_dataset
    from model import ImageCaptionModel
    from train_step import train_step
    from loss import loss_function, compute_loss_components, aggregate_validation_loss
    from save_tokenizer import save_tokenizer
    from evaluate import compute_split_manifest_fingerprint, save_experiment_config_safely

DEFAULT_SPLITS_FILE = REPO_ROOT / "splits.json"
HISTORICAL_MODELS_DIR = (V2_ROOT / "models").resolve()


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
    parser = argparse.ArgumentParser(description="Train V2 Spatial Attention Captioning Model.")
    parser.add_argument("--split", type=str, default="canonical", choices=["canonical", "historical"],
                        help="Dataset split to train on (default: canonical)")
    parser.add_argument("--splits_file", type=str, default=str(DEFAULT_SPLITS_FILE),
                        help="Path to splits.json manifest")
    parser.add_argument("--output_dir", type=str, default=str(V2_ROOT / "experiments" / "v2_canonical"),
                        help="Output directory for new experiment artifacts")
    parser.add_argument("--features_file", type=str, default=None,
                        help="Path to precomputed features .npy (optional)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--min_frequency", type=int, default=5, help="Minimum token frequency for vocabulary")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Word embedding dimension")
    parser.add_argument("--units", type=int, default=256, help="Decoder RNN units")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for algorithmic reproducibility (default: 42)")
    parser.add_argument("--dry_run", action="store_true", help="Audit setup and build model without training")
    parser.add_argument("--allow_overwrite_historical", action="store_true",
                        help="Explicit flag required to overwrite historical models/ directory")
    parser.add_argument("--allow_overwrite_experiment", action="store_true",
                        help="Explicit flag required to overwrite an existing completed experiment directory")
    return parser.parse_args(args)


@tf.function
def validation_step(
    model,
    image_features,
    captions,
    hidden,
    cell
):
    predictions = model(
        image_features,
        captions,
        hidden,
        cell
    )
    targets = captions[:, 1:]
    return compute_loss_components(targets, predictions)


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
        split_info = splits_data.get("historical_v2_split", {})
        train_images = split_info.get("train", [])
        val_images = split_info.get("val", [])
        test_images = []
        assert set(train_images).isdisjoint(set(val_images)), "Historical train and val overlap!"
        print("\nHistorical V2 Split Loaded:")
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
    if args.units <= 0:
        raise ValueError(f"units must be > 0, got {args.units}")
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


def train_v2():
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
            f"Specify a separate experiment output directory (e.g., --output_dir experiments/v2_canonical)."
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
    model = ImageCaptionModel(
        embedding_dim=args.embedding_dim,
        units=args.units,
        vocab_size=vocab_size
    )

    # Build model graph with dummy tensors
    dummy_feat = tf.zeros((1, 64, 2048), dtype=tf.float32)
    dummy_cap = tf.ones((1, max_length), dtype=tf.int32)
    h_init, c_init = model.decoder.reset_state(1)
    model(dummy_feat, dummy_cap, h_init, c_init)
    print("Model initialized and verified. Parameter shapes confirmed:")
    print(f"  Embedding: {model.decoder.embedding.input_dim} -> {model.decoder.embedding.output_dim}")
    print(f"  Output FC: {model.decoder.fc2.units} units")

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
        "model_version": "V2_Attention",
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
        "checkpoint_file": "models/v2_attention.weights.h5",
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
        "units": args.units,
        "min_frequency": args.min_frequency,
        "architecture": {
            "embedding_dim": args.embedding_dim,
            "units": args.units,
            "min_frequency": args.min_frequency
        },
        "loss_objective": "token_normalized_masked_sparse_categorical_crossentropy",
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

    # Check for feature file
    features_dict = None
    if args.features_file and Path(args.features_file).exists():
        print(f"\nLoading precomputed features from: {args.features_file}")
        features_dict = load_features(args.features_file)
    else:
        raise FileNotFoundError(
            "Feature file not specified or not found. Provide --features_file path to run full training."
        )

    print("\nBuilding streaming datasets...")
    train_dataset = build_training_dataset(
        train_captions,
        features_dict,
        word_to_index,
        max_length,
        args.batch_size,
        seed=args.seed
    )
    val_dataset = build_training_dataset(
        val_captions,
        features_dict,
        word_to_index,
        max_length,
        args.batch_size,
        shuffle=False
    )

    best_val_loss = float("inf")
    history_log = {"train_loss": [], "val_loss": []}

    print("\n========================================")
    print("Starting Model Training")
    print("========================================\n")

    for epoch in range(args.epochs):
        start = time.time()
        total_train_loss = 0.0
        train_batches = 0

        for img_feat, caps in train_dataset:
            bs = tf.shape(img_feat)[0]
            hidden, cell = model.decoder.reset_state(bs)
            loss = train_step(model, img_feat, caps, hidden, cell)
            total_train_loss += loss
            train_batches += 1

        train_loss = float(total_train_loss / train_batches)

        total_val_loss_sum = 0.0
        total_val_token_count = 0.0
        val_batches = 0
        for img_feat, caps in val_dataset:
            bs = tf.shape(img_feat)[0]
            hidden, cell = model.decoder.reset_state(bs)
            batch_loss_sum, batch_token_count = validation_step(model, img_feat, caps, hidden, cell)
            total_val_loss_sum += float(batch_loss_sum)
            total_val_token_count += float(batch_token_count)
            val_batches += 1

        # Dataset-level validation loss is globally token-weighted across all non-padding target tokens;
        # fails closed if validation dataset is empty, total valid tokens == 0, or loss is non-finite.
        val_loss = aggregate_validation_loss(total_val_loss_sum, total_val_token_count, num_batches=val_batches)
        elapsed = time.time() - start

        history_log["train_loss"].append(train_loss)
        history_log["val_loss"].append(val_loss)

        # Atomic history save after each completed epoch
        save_history_safely(history_log, output_dir / "history.json")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss (mean batch): {train_loss:.4f} | Val Loss (global token): {val_loss:.4f} | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights_save_path = output_models_dir / "v2_attention.weights.h5"
            model.save_weights(str(weights_save_path))
            print("  ✓ Best model weights saved.")
            # Atomic metadata update with new best checkpoint fingerprint
            if weights_save_path.exists():
                config_data["checkpoint_fingerprint"] = compute_file_sha256(weights_save_path)
                config_data["completed_epochs"] = epoch + 1
                config_data["best_val_loss"] = float(val_loss)
                save_experiment_config_safely(config_data, output_dir / "config.json")

    save_history_safely(history_log, output_dir / "history.json")

    # Record saved best checkpoint fingerprint in experiment metadata
    best_wt_file = output_models_dir / "v2_attention.weights.h5"
    if best_wt_file.exists():
        config_data["checkpoint_fingerprint"] = compute_file_sha256(best_wt_file)
        config_data["completed_epochs"] = len(history_log["train_loss"])
        if history_log["val_loss"]:
            config_data["best_val_loss"] = float(min(history_log["val_loss"]))
        save_experiment_config_safely(config_data, output_dir / "config.json")

    print("\nTraining complete.")
    print("Best weights:", output_models_dir / "v2_attention.weights.h5")


if __name__ == "__main__":
    train_v2()
