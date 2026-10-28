
"""Fine-tune BLIP on Flickr30k using PyTorch.

The default settings run a small pipeline smoke test.
Use --max-train-rows 0 and --max-val-rows 0 to use full splits.
"""

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    get_linear_schedule_with_warmup,
)

from src.dataset import Flickr30kCaptionDataset, BLIPCaptionCollator


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "flickr30k"
MODEL_NAME = "Salesforce/blip-image-captioning-base"


def set_seed(seed: int) -> None:
    """Set random seeds for more reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_configuration(device, no_amp):
    """Choose a suitable mixed-precision mode for this device."""
    use_amp = device.type == "cuda" and not no_amp

    if not use_amp:
        return False, torch.float32, False

    if torch.cuda.is_bf16_supported():
        # BF16 generally does not require gradient scaling.
        return True, torch.bfloat16, False

    # FP16 uses GradScaler to reduce gradient underflow.
    return True, torch.float16, True


def autocast_context(use_amp, amp_dtype):
    """Return the appropriate autocast context."""
    if not use_amp:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    )


def move_batch_to_device(batch, device):
    """Move tensor values to the selected compute device."""
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def evaluate_loss(model, loader, device, use_amp, amp_dtype):
    """Calculate caption token loss on the validation set."""
    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            with autocast_context(use_amp, amp_dtype):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

            batch_size = batch["pixel_values"].size(0)
            total_loss += outputs.loss.float().item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("Validation dataloader is empty.")

    return total_loss / total_examples


def make_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained BLIP on Flickr30k."
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=40)

    # Small defaults make the first run a pipeline smoke test.
    # These values refer to caption rows, not unique images.
    parser.add_argument("--max-train-rows", type=int, default=32)
    parser.add_argument("--max-val-rows", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision training.",
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2.")
    if args.max_train_rows < 0 or args.max_val_rows < 0:
        parser.error("Dataset limits must be zero or positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp, amp_dtype, use_scaler = get_amp_configuration(
        device, args.no_amp
    )

    print("=" * 60)
    print("BLIP FLICKR30K FINE-TUNING")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {properties.total_memory / 1024**3:.2f} GB")

    print(f"Checkpoint initialization: {MODEL_NAME}")
    print(f"Mixed precision: {use_amp}")
    print(
        "AMP dtype:",
        str(amp_dtype) if use_amp else "float32",
    )
    print(f"Gradient scaler enabled: {use_scaler}")

    print("\nLoading processor and pretrained model...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    train_dataset = Flickr30kCaptionDataset(
        DATA_ROOT / "train",
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    val_dataset = Flickr30kCaptionDataset(
        DATA_ROOT / "validation",
        max_rows=args.max_val_rows,
        seed=args.seed + 1,
    )

    collator = BLIPCaptionCollator(
        processor=processor,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    accumulation = args.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = min(10, total_updates // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    run_dir = ROOT / "results" / "training"
    checkpoint_dir = ROOT / "checkpoints" / "flickr30k_blip"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "training_history.csv"
    history = []
    best_val_loss = float("inf")
    global_update = 0

    print("\nRun configuration")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {accumulation}")
    print(f"Training caption pairs: {len(train_dataset):,}")
    print(f"Validation caption pairs: {len(val_dataset):,}")
    print(f"Optimizer updates planned: {total_updates:,}")
    print("\nStarting fine-tuning...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_train_loss = 0.0
        total_train_examples = 0
        skipped_updates = 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            batch_size = batch["pixel_values"].size(0)

            # Handle a final, incomplete gradient-accumulation group
            # without unnecessarily shrinking its gradient.
            group_start = (step // accumulation) * accumulation
            group_size = min(
                accumulation,
                len(train_loader) - group_start,
            )

            with autocast_context(use_amp, amp_dtype):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                raw_loss = outputs.loss
                loss = raw_loss / group_size

            if not torch.isfinite(raw_loss.detach()):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, "
                    f"batch {step + 1}: {raw_loss.detach().item()}"
                )

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_train_loss += raw_loss.detach().float().item() * batch_size
            total_train_examples += batch_size

            is_update_step = (
                (step + 1) % accumulation == 0
                or (step + 1) == len(train_loader)
            )

            if is_update_step:
                if use_scaler:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                # Only advance the scheduler if an optimizer update
                # actually occurred. GradScaler can skip an FP16 update
                # when it detects overflow.
                if use_scaler:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_succeeded = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()
                    step_succeeded = True

                optimizer.zero_grad(set_to_none=True)

                if step_succeeded:
                    scheduler.step()
                    global_update += 1
                else:
                    skipped_updates += 1
                    print(
                        "Warning: FP16 overflow; optimizer and "
                        "scheduler update skipped."
                    )

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {step + 1}/{len(train_loader)} | "
                    f"Loss {raw_loss.detach().float().item():.4f}"
                )

        train_loss = total_train_loss / max(total_train_examples, 1)

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            use_amp,
            amp_dtype,
        )

        elapsed = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "successful_optimizer_updates": global_update,
            "skipped_optimizer_updates": skipped_updates,
        }
        history.append(record)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{args.epochs} complete")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Successful optimizer updates: {global_update}")
        print(f"Skipped optimizer updates: {skipped_updates}")
        print(f"Epoch time: {elapsed:.1f} seconds")
        print("=" * 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)

            print(f"Best model checkpoint saved: {checkpoint_dir}")

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

    config = {
        "initialization_checkpoint": MODEL_NAME,
        "dataset": "Flickr30k",
        "training_mode": "fine-tuning",
        "note": (
            "The initialization checkpoint has already been fine-tuned "
            "for COCO captioning. This experiment fine-tunes it on "
            "Flickr30k; it is not pretraining from scratch."
        ),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "mixed_precision": use_amp,
        "amp_dtype": str(amp_dtype) if use_amp else "float32",
        "gradient_scaler_enabled": use_scaler,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "num_workers": args.num_workers,
        "best_validation_loss": best_val_loss,
        "successful_optimizer_updates": global_update,
    }

    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\nTraining run finished successfully.")
    print(f"Training history: {history_path}")
    print(f"Run configuration: {config_path}")
    print(f"Best checkpoint: {checkpoint_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
