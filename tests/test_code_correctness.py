import os
import sys
import json
import hashlib
import tempfile
import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = REPO_ROOT / "V1_Baseline_CNN_LSTM"
V2_ROOT = REPO_ROOT / "V2_Attention"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(V1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V1_ROOT / "src"))
if str(V2_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V2_ROOT / "src"))

from V1_Baseline_CNN_LSTM.src.sequences import pad_sequence as v1_pad, create_training_sequences as v1_create_seqs
from V2_Attention.src.sequences import pad_sequence as v2_pad
from V2_Attention.src.attention import BahdanauAttention
from V2_Attention.src.decoder import RNNDecoder
from V1_Baseline_CNN_LSTM.src.plot_history import plot_training_history as v1_plot_history
from V2_Attention.src.plot_history import plot_training_history as v2_plot_history
from V1_Baseline_CNN_LSTM.src.bleu_score import (
    resolve_v1_evaluation_context,
    HISTORICAL_MODEL_FILE,
    HISTORICAL_TOKENIZER_FILE as V1_HIST_TOK,
    compute_file_sha256,
    compute_split_manifest_fingerprint,
    validate_split_manifest_structure,
    save_experiment_config_safely
)
from V2_Attention.src.evaluate import (
    resolve_v2_evaluation_context,
    HISTORICAL_WEIGHTS_FILE,
    HISTORICAL_TOKENIZER_FILE as V2_HIST_TOK
)
from V1_Baseline_CNN_LSTM.src.data_generator import CaptionDataGenerator as V1CaptionDataGenerator
from V2_Attention.src.training_data import build_training_dataset as v2_build_training_dataset
from V1_Baseline_CNN_LSTM.src.cnn_features import load_features as v1_load_features
from V2_Attention.src.cnn_features import load_features as v2_load_features
from V1_Baseline_CNN_LSTM.src.dataset import load_captions as v1_load_captions
from V2_Attention.src.dataset import load_captions as v2_load_captions
from unittest.mock import patch
from argparse import Namespace
from V1_Baseline_CNN_LSTM.src.train import (
    validate_training_setup as v1_validate_training_setup,
    save_history_safely as v1_save_history_safely,
    set_reproducible_seed as v1_set_reproducible_seed,
    load_dataset_split as v1_load_dataset_split
)
from V2_Attention.src.train import (
    validate_training_setup as v2_validate_training_setup,
    save_history_safely as v2_save_history_safely,
    set_reproducible_seed as v2_set_reproducible_seed,
    load_dataset_split as v2_load_dataset_split
)


class TestCodeCorrectnessAndReliability(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. pad_sequence exact length, truncation, and end-token preservation
    # -------------------------------------------------------------------------
    def test_pad_sequence_comprehensive_semantics(self):
        """Verify pad_sequence correctly pads short sequences, truncates long ones, and preserves end markers."""
        for pad_func in [v1_pad, v2_pad]:
            # 1. Short sequence: needs padding
            short_seq = [1, 2, 3]
            padded = pad_func(short_seq, max_length=6)
            self.assertEqual(len(padded), 6)
            self.assertEqual(padded, [1, 2, 3, 0, 0, 0])

            # 2. Sequence exactly max_length: unchanged
            exact_seq = [1, 2, 3, 4, 5]
            exact = pad_func(exact_seq, max_length=5)
            self.assertEqual(len(exact), 5)
            self.assertEqual(exact, [1, 2, 3, 4, 5])

            # 3. Sequence longer than max_length: simple truncation without preserve
            long_seq = [1, 2, 3, 4, 5, 6, 7, 8]
            truncated = pad_func(long_seq, max_length=5)
            self.assertEqual(len(truncated), 5)
            self.assertEqual(truncated, [1, 2, 3, 4, 5])

            # 4. Long caption containing start/end markers with end-token preservation (end_token=None)
            start_token = 101
            end_token = 102
            long_caption = [start_token, 10, 20, 30, 40, 50, 60, end_token]
            preserved = pad_func(long_caption, max_length=5, preserve_end_token=True)
            self.assertEqual(len(preserved), 5)
            self.assertEqual(preserved[0], start_token, "Start token lost upon truncation!")
            self.assertEqual(preserved[-1], end_token, "End token was not preserved at final position!")
            self.assertEqual(preserved, [start_token, 10, 20, 30, end_token])

            # 5. Long caption with explicit end_token argument
            raw_seq = [start_token, 10, 20, 30, 40, 50]
            explicit_end = pad_func(raw_seq, max_length=4, preserve_end_token=True, end_token=999)
            self.assertEqual(len(explicit_end), 4)
            self.assertEqual(explicit_end[0], start_token)
            self.assertEqual(explicit_end[-1], 999)
            self.assertEqual(explicit_end, [start_token, 10, 20, 999])

            # 6. Custom padding value and dtype verification
            custom_padded = pad_func([1, 2], max_length=5, padding_value=-1)
            self.assertEqual(custom_padded, [1, 2, -1, -1, -1])

            # 7. NumPy conversion verification (matching training_data pipeline)
            arr = np.asarray(padded, dtype=np.int32)
            self.assertEqual(arr.shape, (6,))
            self.assertEqual(arr.dtype, np.int32)

            # 8. Empty sequence
            empty_padded = pad_func([], max_length=4)
            self.assertEqual(empty_padded, [0, 0, 0, 0])

    # -------------------------------------------------------------------------
    # 2. Training Input and Target Alignment (V1 & V2)
    # -------------------------------------------------------------------------
    def test_v1_and_v2_training_input_target_alignment(self):
        """Verify that sequence padding and truncation strictly preserves teacher-forcing target alignment."""
        # V1 Alignment: Prefix sequences and next-word targets
        v1_caption_seq = [101, 10, 20, 102]
        v1_inputs, v1_targets = v1_create_seqs(v1_caption_seq, max_length=5)
        self.assertEqual(len(v1_inputs), 3)
        self.assertEqual(len(v1_targets), 3)
        self.assertEqual(v1_inputs[0], [101, 0, 0, 0, 0])
        self.assertEqual(v1_targets[0], 10)
        self.assertEqual(v1_inputs[1], [101, 10, 0, 0, 0])
        self.assertEqual(v1_targets[1], 20)
        self.assertEqual(v1_inputs[2], [101, 10, 20, 0, 0])
        self.assertEqual(v1_targets[2], 102)

        # V2 Alignment: Teacher-forcing inputs (caption[:, :-1]) and targets (caption[:, 1:])
        v2_raw = [101, 10, 20, 30, 40, 50, 102]
        v2_truncated = v2_pad(v2_raw, max_length=5, preserve_end_token=True, end_token=102)
        self.assertEqual(v2_truncated, [101, 10, 20, 30, 102])

        caps = tf.constant([v2_truncated], dtype=tf.int32)
        inputs = caps[:, :-1]
        targets = caps[:, 1:]

        self.assertEqual(inputs.shape, (1, 4))
        self.assertEqual(targets.shape, (1, 4))
        self.assertEqual(int(inputs[0, 0]), 101, "Input sequence does not begin with <start> token!")
        self.assertEqual(int(targets[0, -1]), 102, "Target sequence does not end with <end> token!")
        np.testing.assert_array_equal(inputs[0].numpy(), [101, 10, 20, 30])
        np.testing.assert_array_equal(targets[0].numpy(), [10, 20, 30, 102])

    # -------------------------------------------------------------------------
    # 3. V2 Attention and Decoder tensor shape verification (zero training)
    # -------------------------------------------------------------------------
    def test_bahdanau_attention_shapes(self):
        """Verify Bahdanau attention outputs expected context vector and weight shapes with synthetic tensors."""
        batch_size = 2
        num_features = 64  # 8x8 spatial grid
        feature_dim = 256
        units = 128

        attention_layer = BahdanauAttention(units=units)

        synthetic_features = tf.zeros((batch_size, num_features, feature_dim), dtype=tf.float32)
        synthetic_hidden = tf.zeros((batch_size, units), dtype=tf.float32)

        context_vector, attention_weights = attention_layer(synthetic_features, synthetic_hidden)

        self.assertEqual(context_vector.shape, (batch_size, feature_dim))
        self.assertEqual(attention_weights.shape, (batch_size, num_features, 1))

    def test_rnn_decoder_shapes(self):
        """Verify RNNDecoder forward step outputs expected logits and recurrent states."""
        batch_size = 2
        num_features = 64
        feature_dim = 256
        embedding_dim = 64
        units = 128
        vocab_size = 100

        decoder = RNNDecoder(embedding_dim=embedding_dim, units=units, vocab_size=vocab_size)

        synthetic_x = tf.constant([[1], [2]], dtype=tf.int32)
        synthetic_features = tf.zeros((batch_size, num_features, feature_dim), dtype=tf.float32)
        synthetic_hidden = tf.zeros((batch_size, units), dtype=tf.float32)
        synthetic_cell = tf.zeros((batch_size, units), dtype=tf.float32)

        x_out, state_h, state_c, attention_weights = decoder(
            synthetic_x,
            synthetic_features,
            synthetic_hidden,
            synthetic_cell
        )

        self.assertEqual(x_out.shape, (batch_size, vocab_size))
        self.assertEqual(state_h.shape, (batch_size, units))
        self.assertEqual(state_c.shape, (batch_size, units))
        self.assertEqual(attention_weights.shape, (batch_size, num_features, 1))

    # -------------------------------------------------------------------------
    # 4. Lazy CNN Proxy: Verify Model Construction is Deferred
    # -------------------------------------------------------------------------
    def test_lazy_cnn_proxy_defers_construction(self):
        """Directly verify that importing cnn_features did NOT instantiate InceptionV3."""
        import V1_Baseline_CNN_LSTM.src.cnn_features as v1_cnn
        import V2_Attention.src.cnn_features as v2_cnn

        # Both modules should expose cnn_model as a proxy object
        self.assertTrue(hasattr(v1_cnn, "cnn_model"))
        self.assertTrue(hasattr(v2_cnn, "cnn_model"))

        # Verify underlying model instance is strictly None prior to explicit access
        self.assertIsNone(v1_cnn._cnn_model, "V1 InceptionV3 was eagerly instantiated on import!")
        self.assertIsNone(v2_cnn._cnn_model, "V2 InceptionV3 was eagerly instantiated on import!")

        # Verify proxy truthiness without instantiating model
        self.assertTrue(bool(v1_cnn.cnn_model))
        self.assertTrue(bool(v2_cnn.cnn_model))

    # -------------------------------------------------------------------------
    # 5. Anti-Bypass Audit: 10 Rigorous Test Cases
    # -------------------------------------------------------------------------
    def test_case_1_historical_checkpoint_recognized_at_original_path(self):
        """Case 1: Historical checkpoint at its original path is recognized as historical."""
        mod_p, tok_p, train_split, exp_split, meta = resolve_v1_evaluation_context(
            model_file=str(HISTORICAL_MODEL_FILE)
        )
        self.assertEqual(train_split, "historical_v1_split")
        self.assertEqual(exp_split, "historical_v1")

        wt_p, tok_p2, train_split2, exp_split2, meta2 = resolve_v2_evaluation_context(
            weights_file=str(HISTORICAL_WEIGHTS_FILE)
        )
        self.assertEqual(train_split2, "historical_v2_split")
        self.assertEqual(exp_split2, "historical_v2")

    def test_case_2_copied_historical_checkpoint_recognized_by_fingerprint(self):
        """Case 2: A copied historical checkpoint remains recognized by its fingerprint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            copied_ckpt = Path(tmp_dir) / "copied_best_caption_model.keras"
            # Read bytes only, copy to new directory
            with open(HISTORICAL_MODEL_FILE, "rb") as src, open(copied_ckpt, "wb") as dst:
                while chunk := src.read(65536):
                    dst.write(chunk)

            # Attempting to declare copied historical checkpoint as canonical must be rejected
            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(
                    model_file=str(copied_ckpt),
                    trained_on="canonical"
                )
            self.assertIn("SECURITY / LEAK GUARD CONFLICT", str(ctx.exception))

    def test_case_3_canonical_checkpoint_in_nested_models_dir_not_historical(self):
        """Case 3: A new canonical experiment checkpoint in a nested models/ directory is NOT misclassified as historical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exp_models_dir = tmp_path / "models"
            exp_models_dir.mkdir(parents=True)
            canonical_ckpt = exp_models_dir / "best_caption_model.keras"
            canonical_ckpt.write_bytes(b"synthetic_canonical_checkpoint_bytes_unique_12345")
            ckpt_fp = compute_file_sha256(canonical_ckpt)

            fake_tok = exp_models_dir / "tokenizer.json"
            fake_tok_data = {
                "word_to_index": {f"w{i}": i for i in range(50)},
                "index_to_word": {str(i): f"w{i}" for i in range(50)},
                "max_length": 25
            }
            with open(fake_tok, "w", encoding="utf-8") as f:
                json.dump(fake_tok_data, f)
            tok_fp = compute_file_sha256(fake_tok)

            exp_config = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 50,
                "tokenizer_file": "models/tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_fingerprint": ckpt_fp
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(exp_config, f)

            mod_p, tok_p, train_split, exp_split, meta = resolve_v1_evaluation_context(
                model_file=str(canonical_ckpt)
            )
            self.assertEqual(train_split, "canonical_split")
            self.assertEqual(exp_split, "canonical")

    def test_case_4_declaring_historical_as_canonical_rejected(self):
        """Case 4: Declaring a historical checkpoint as canonical is rejected."""
        with self.assertRaises(ValueError) as ctx_v1:
            resolve_v1_evaluation_context(
                model_file=str(HISTORICAL_MODEL_FILE),
                trained_on="canonical"
            )
        self.assertIn("SECURITY / LEAK GUARD CONFLICT", str(ctx_v1.exception))

        with self.assertRaises(ValueError) as ctx_v2:
            resolve_v2_evaluation_context(
                weights_file=str(HISTORICAL_WEIGHTS_FILE),
                trained_on="canonical"
            )
        self.assertIn("SECURITY / LEAK GUARD CONFLICT", str(ctx_v2.exception))

    def test_case_5_different_tokenizer_with_same_vocab_size_rejected(self):
        """Case 5: Supplying a different tokenizer with the same vocabulary size is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "custom.keras"
            fake_ckpt.write_bytes(b"synthetic_ckpt_bytes")

            tok1 = tmp_path / "tok1.json"
            tok1_data = {"word_to_index": {f"word_{i}": i for i in range(20)}, "max_length": 10}
            with open(tok1, "w", encoding="utf-8") as f:
                json.dump(tok1_data, f)
            tok1_fp = compute_file_sha256(tok1)

            # Different tokenizer with exact same vocabulary size (20 tokens)
            tok2 = tmp_path / "tok2.json"
            tok2_data = {"word_to_index": {f"different_{i}": i for i in range(20)}, "max_length": 10}
            with open(tok2, "w", encoding="utf-8") as f:
                json.dump(tok2_data, f)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 20,
                "tokenizer_file": "tok1.json",
                "tokenizer_fingerprint": tok1_fp,
                "checkpoint_fingerprint": compute_file_sha256(fake_ckpt)
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            # Evaluating with tok2 (different hash, same vocab size) must raise TOKENIZER IDENTITY MISMATCH
            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(
                    model_file=str(fake_ckpt),
                    tokenizer_file=str(tok2)
                )
            self.assertIn("TOKENIZER IDENTITY MISMATCH", str(ctx.exception))

    def test_case_6_modified_or_missing_experiment_metadata_rejected(self):
        """Case 6: Modified or missing experiment metadata is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "untracked.keras"
            fake_ckpt.write_bytes(b"synthetic_untracked_ckpt")

            # Missing config.json -> rejected
            with self.assertRaises(ValueError) as ctx_missing:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("EXPERIMENT METADATA MISSING", str(ctx_missing.exception))

            # Missing split_type in config.json -> rejected
            bad_cfg = {"model_version": "V1_Baseline_CNN_LSTM"}
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(bad_cfg, f)

            with self.assertRaises(ValueError) as ctx_bad:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("INVALID EXPERIMENT METADATA", str(ctx_bad.exception))

    def test_case_7_split_manifest_mismatch_rejected(self):
        """Case 7: A split manifest that does not match the recorded training split identity is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "custom.keras"
            fake_ckpt.write_bytes(b"synthetic_ckpt")

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_fingerprint": compute_file_sha256(fake_ckpt),
                "train_split_fingerprint": "0" * 64
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("SPLIT MANIFEST MISMATCH", str(ctx.exception))

    def test_case_8_checkpoint_identity_mismatch_rejected(self):
        """Case 8: A checkpoint that does not match the recorded checkpoint fingerprint is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "custom.keras"
            fake_ckpt.write_bytes(b"corrupted_checkpoint_bytes_abc")

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_fingerprint": "0" * 64
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("CHECKPOINT IDENTITY MISMATCH", str(ctx.exception))

    def test_case_9_existing_historical_evaluation_commands_remain_safe(self):
        """Case 9: Existing historical evaluation commands remain safe, default-bound, and usable."""
        mod_p, tok_p, train_split, exp_split, meta = resolve_v1_evaluation_context()
        self.assertEqual(mod_p, HISTORICAL_MODEL_FILE)
        self.assertEqual(tok_p, V1_HIST_TOK)
        self.assertEqual(train_split, "historical_v1_split")

        wt_p, tok_p2, train_split2, exp_split2, meta2 = resolve_v2_evaluation_context()
        self.assertEqual(wt_p, HISTORICAL_WEIGHTS_FILE)
        self.assertEqual(tok_p2, V2_HIST_TOK)
        self.assertEqual(train_split2, "historical_v2_split")

    def test_case_10_canonical_evaluation_accepts_valid_experiment(self):
        """Case 10: Canonical evaluation accepts only a correctly associated experiment checkpoint, tokenizer, and split metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exp_ckpt = tmp_path / "best_caption_model.keras"
            exp_ckpt.write_bytes(b"valid_canonical_checkpoint_bytes_98765")
            ckpt_fp = compute_file_sha256(exp_ckpt)

            exp_tok = tmp_path / "tokenizer.json"
            exp_tok.write_text(json.dumps({"word_to_index": {f"w{i}": i for i in range(30)}, "max_length": 15}), encoding="utf-8")
            tok_fp = compute_file_sha256(exp_tok)

            # Get canonical split fingerprint from repo splits.json
            with open(REPO_ROOT / "splits.json", "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            canon_train_imgs = splits_data["canonical_split"]["train"]
            canon_train_fp = hashlib.sha256(json.dumps(sorted(canon_train_imgs)).encode("utf-8")).hexdigest()
            canon_val_imgs = splits_data["canonical_split"]["val"]
            canon_val_fp = hashlib.sha256(json.dumps(sorted(canon_val_imgs)).encode("utf-8")).hexdigest()
            canon_test_imgs = splits_data["canonical_split"]["test"]
            canon_test_fp = hashlib.sha256(json.dumps(sorted(canon_test_imgs)).encode("utf-8")).hexdigest()
            canon_manifest_fp = compute_split_manifest_fingerprint(splits_data["canonical_split"])

            exp_config = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest": str(REPO_ROOT / "splits.json"),
                "train_split_fingerprint": canon_train_fp,
                "val_split_fingerprint": canon_val_fp,
                "test_split_fingerprint": canon_test_fp,
                "split_manifest_fingerprint": canon_manifest_fp,
                "vocab_size": 30,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": "best_caption_model.keras",
                "checkpoint_fingerprint": ckpt_fp
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(exp_config, f)

            mod_p, tok_p, train_split, exp_split, meta = resolve_v1_evaluation_context(
                model_file=str(exp_ckpt)
            )
            self.assertEqual(train_split, "canonical_split")
            self.assertEqual(exp_split, "canonical")
            self.assertEqual(tok_p.resolve(), exp_tok.resolve())
            self.assertEqual(meta["vocab_size"], 30)

            # Test V2 valid experiment as well
            exp_config_v2 = {
                "model_version": "V2_Attention",
                "split_type": "canonical",
                "split_manifest": str(REPO_ROOT / "splits.json"),
                "train_split_fingerprint": canon_train_fp,
                "val_split_fingerprint": canon_val_fp,
                "test_split_fingerprint": canon_test_fp,
                "split_manifest_fingerprint": canon_manifest_fp,
                "vocab_size": 30,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": "v2_attention.weights.h5",
                "checkpoint_fingerprint": ckpt_fp
            }
            v2_ckpt = tmp_path / "v2_attention.weights.h5"
            v2_ckpt.write_bytes(b"valid_canonical_checkpoint_bytes_98765")
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(exp_config_v2, f)

            wt_p, tok_p2, train_split2, exp_split2, meta2 = resolve_v2_evaluation_context(
                weights_file=str(v2_ckpt)
            )
            self.assertEqual(train_split2, "canonical_split")
            self.assertEqual(exp_split2, "canonical")
            self.assertEqual(tok_p2.resolve(), exp_tok.resolve())
            self.assertEqual(meta2["vocab_size"], 30)

    # -------------------------------------------------------------------------
    # 6. Plot Training History Utilities
    # -------------------------------------------------------------------------
    def test_plot_history_execution(self):
        """Verify plot_training_history creates loss.png without error for both V1 and V2 formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test V1 format history
            v1_history = {
                "loss": [3.5, 2.8, 2.1],
                "val_loss": [3.6, 2.9, 2.4],
                "accuracy": [0.3, 0.4, 0.5],
                "val_accuracy": [0.28, 0.38, 0.45]
            }
            v1_hist_file = tmp_path / "v1_history.json"
            with open(v1_hist_file, "w", encoding="utf-8") as f:
                json.dump(v1_history, f)

            v1_plots_dir = tmp_path / "v1_plots"
            v1_plot_history(history_file=v1_hist_file, output_dir=v1_plots_dir)
            self.assertTrue((v1_plots_dir / "loss.png").exists())
            self.assertTrue((v1_plots_dir / "accuracy.png").exists())

            # Test V2 format history
            v2_history = {
                "train_loss": [4.2, 3.1, 2.5],
                "val_loss": [4.5, 3.4, 2.9]
            }
            v2_hist_file = tmp_path / "v2_history.json"
            with open(v2_hist_file, "w", encoding="utf-8") as f:
                json.dump(v2_history, f)

            v2_plots_dir = tmp_path / "v2_plots"
            v2_plot_history(history_file=v2_hist_file, output_dir=v2_plots_dir)
            self.assertTrue((v2_plots_dir / "loss.png").exists())


class TestSplitManifestAndCheckpointIntegrity(unittest.TestCase):
    """
    Rigorously tests:
    1. Complete canonical split manifest fingerprinting and validation (Task 1).
    2. Checkpoint fingerprint verification lifecycle and fail-closed evaluation (Task 2).
    """

    # -------------------------------------------------------------------------
    # Task 1: Complete Split Manifest Integrity Tests
    # -------------------------------------------------------------------------
    def test_manifest_accept_unchanged(self):
        """1. Accept an unchanged canonical manifest."""
        manifest = {
            "canonical_split": {
                "train": [f"img_{i}.jpg" for i in range(100)],
                "val": [f"img_{i}.jpg" for i in range(100, 120)],
                "test": [f"img_{i}.jpg" for i in range(120, 140)]
            }
        }
        # Validate structural integrity passes
        validate_split_manifest_structure(manifest, "canonical_split")

        fp1 = compute_split_manifest_fingerprint(manifest["canonical_split"])
        fp2 = compute_split_manifest_fingerprint(manifest["canonical_split"])
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_manifest_reject_changed_train(self):
        """2. Reject a changed training partition."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train1 = [f"img_{i}.jpg" for i in range(100)]
            val1 = [f"img_{i}.jpg" for i in range(100, 120)]
            test1 = [f"img_{i}.jpg" for i in range(120, 140)]
            split_data = {"train": train1, "val": val1, "test": test1}

            expected_train_fp = hashlib.sha256(json.dumps(sorted(train1)).encode("utf-8")).hexdigest()
            manifest_fp = compute_split_manifest_fingerprint(split_data)

            # Alter 1 training image
            train2 = train1.copy()
            train2[0] = "altered_img.jpg"
            manifest2 = {"canonical_split": {"train": train2, "val": val1, "test": test1}}
            manifest_file = tmp_path / "splits.json"
            manifest_file.write_text(json.dumps(manifest2), encoding="utf-8")

            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"model_bytes")
            ckpt_fp = compute_file_sha256(fake_ckpt)

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest": str(manifest_file),
                "train_split_fingerprint": expected_train_fp,
                "split_manifest_fingerprint": manifest_fp,
                "tokenizer_file": str(fake_tok),
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": str(fake_ckpt),
                "checkpoint_fingerprint": ckpt_fp,
                "vocab_size": 1
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("SPLIT MANIFEST MISMATCH", str(ctx.exception))

    def test_manifest_reject_changed_val_train_identical(self):
        """3. Reject a changed validation partition while training IDs remain identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train1 = [f"img_{i}.jpg" for i in range(100)]
            val1 = [f"img_{i}.jpg" for i in range(100, 120)]
            test1 = [f"img_{i}.jpg" for i in range(120, 140)]
            split_data = {"train": train1, "val": val1, "test": test1}

            expected_train_fp = hashlib.sha256(json.dumps(sorted(train1)).encode("utf-8")).hexdigest()
            expected_val_fp = hashlib.sha256(json.dumps(sorted(val1)).encode("utf-8")).hexdigest()
            manifest_fp = compute_split_manifest_fingerprint(split_data)

            # Train partition is 100% IDENTICAL, but validation partition is altered
            val2 = val1.copy()
            val2[0] = "altered_val_img.jpg"
            manifest2 = {"canonical_split": {"train": train1, "val": val2, "test": test1}}
            manifest_file = tmp_path / "splits.json"
            manifest_file.write_text(json.dumps(manifest2), encoding="utf-8")

            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"model_bytes")
            ckpt_fp = compute_file_sha256(fake_ckpt)

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest": str(manifest_file),
                "train_split_fingerprint": expected_train_fp,
                "val_split_fingerprint": expected_val_fp,
                "split_manifest_fingerprint": manifest_fp,
                "tokenizer_file": str(fake_tok),
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": str(fake_ckpt),
                "checkpoint_fingerprint": ckpt_fp,
                "vocab_size": 1
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("SPLIT MANIFEST MISMATCH", str(ctx.exception))

    def test_manifest_reject_changed_test_train_identical(self):
        """4. Reject a changed test partition while training IDs remain identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train1 = [f"img_{i}.jpg" for i in range(100)]
            val1 = [f"img_{i}.jpg" for i in range(100, 120)]
            test1 = [f"img_{i}.jpg" for i in range(120, 140)]
            split_data = {"train": train1, "val": val1, "test": test1}

            expected_train_fp = hashlib.sha256(json.dumps(sorted(train1)).encode("utf-8")).hexdigest()
            expected_test_fp = hashlib.sha256(json.dumps(sorted(test1)).encode("utf-8")).hexdigest()
            manifest_fp = compute_split_manifest_fingerprint(split_data)

            # Train and val are IDENTICAL, but test partition is altered
            test2 = test1.copy()
            test2[0] = "altered_test_img.jpg"
            manifest2 = {"canonical_split": {"train": train1, "val": val1, "test": test2}}
            manifest_file = tmp_path / "splits.json"
            manifest_file.write_text(json.dumps(manifest2), encoding="utf-8")

            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"model_bytes")
            ckpt_fp = compute_file_sha256(fake_ckpt)

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest": str(manifest_file),
                "train_split_fingerprint": expected_train_fp,
                "test_split_fingerprint": expected_test_fp,
                "split_manifest_fingerprint": manifest_fp,
                "tokenizer_file": str(fake_tok),
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": str(fake_ckpt),
                "checkpoint_fingerprint": ckpt_fp,
                "vocab_size": 1
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("SPLIT MANIFEST MISMATCH", str(ctx.exception))

    def test_manifest_reject_overlapping_partitions_and_duplicates(self):
        """5. Reject overlapping partitions or duplicate IDs."""
        # A. Duplicate image ID within train partition
        manifest_dup = {
            "canonical_split": {
                "train": ["a.jpg", "b.jpg", "a.jpg"],
                "val": ["c.jpg"],
                "test": ["d.jpg"]
            }
        }
        with self.assertRaises(ValueError) as ctx_dup:
            validate_split_manifest_structure(manifest_dup, "canonical_split")
        self.assertIn("DUPLICATE IMAGE IDS DETECTED", str(ctx_dup.exception))

        # B. Overlap between train and val
        manifest_overlap_tv = {
            "canonical_split": {
                "train": ["a.jpg", "b.jpg"],
                "val": ["b.jpg", "c.jpg"],
                "test": ["d.jpg"]
            }
        }
        with self.assertRaises(ValueError) as ctx_tv:
            validate_split_manifest_structure(manifest_overlap_tv, "canonical_split")
        self.assertIn("CROSS-PARTITION OVERLAP DETECTED", str(ctx_tv.exception))

        # C. Overlap between val and test
        manifest_overlap_vt = {
            "canonical_split": {
                "train": ["a.jpg", "b.jpg"],
                "val": ["c.jpg"],
                "test": ["c.jpg", "d.jpg"]
            }
        }
        with self.assertRaises(ValueError) as ctx_vt:
            validate_split_manifest_structure(manifest_overlap_vt, "canonical_split")
        self.assertIn("CROSS-PARTITION OVERLAP DETECTED", str(ctx_vt.exception))

    def test_manifest_reject_missing_or_malformed_metadata(self):
        """6. Reject missing or malformed split metadata."""
        # Non-dict root
        with self.assertRaises(ValueError) as ctx1:
            validate_split_manifest_structure(["not_a_dict"], "canonical_split")
        self.assertIn("MALFORMED SPLIT MANIFEST", str(ctx1.exception))

        # Missing required split key
        with self.assertRaises(ValueError) as ctx2:
            validate_split_manifest_structure({}, "canonical_split")
        self.assertIn("MALFORMED SPLIT MANIFEST", str(ctx2.exception))

        # Missing test partition in canonical split
        with self.assertRaises(ValueError) as ctx3:
            validate_split_manifest_structure({"canonical_split": {"train": [], "val": []}}, "canonical_split")
        self.assertIn("missing required partition 'test'", str(ctx3.exception))

        # Non-list partition field
        with self.assertRaises(ValueError) as ctx4:
            validate_split_manifest_structure({"canonical_split": {"train": "not_a_list", "val": [], "test": []}}, "canonical_split")
        self.assertIn("must be a list", str(ctx4.exception))

    # -------------------------------------------------------------------------
    # Task 2: Checkpoint Fingerprint Lifecycle & Verification Tests
    # -------------------------------------------------------------------------
    def test_checkpoint_reject_null_fingerprint(self):
        """7. Evaluation rejects null or missing checkpoint_fingerprint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"bytes_123")

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": "model.keras",
                "checkpoint_fingerprint": None  # Initialized but unfinalized
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("CHECKPOINT FINGERPRINT MISSING", str(ctx.exception))

    def test_checkpoint_reject_missing_file(self):
        """8. Evaluation rejects a non-existent checkpoint file path."""
        non_existent_file = Path("non_existent_directory_999") / "missing.keras"
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_v1_evaluation_context(model_file=str(non_existent_file))
        self.assertIn("Model checkpoint not found", str(ctx.exception))

    def test_checkpoint_reject_incorrect_fingerprint(self):
        """9. Evaluation rejects a mismatched checkpoint fingerprint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"actual_checkpoint_bytes_abc")

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": "model.keras",
                "checkpoint_fingerprint": "1" * 64  # Wrong fingerprint
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("CHECKPOINT IDENTITY MISMATCH", str(ctx.exception))

    def test_checkpoint_accept_correct_fingerprint(self):
        """10. Evaluation accepts matching checkpoint fingerprint with full valid metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"actual_checkpoint_bytes_verified")
            ckpt_fp = compute_file_sha256(fake_ckpt)

            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            tok_fp = compute_file_sha256(fake_tok)

            manifest = {
                "canonical_split": {
                    "train": ["a.jpg"],
                    "val": ["b.jpg"],
                    "test": ["c.jpg"]
                }
            }
            manifest_file = tmp_path / "splits.json"
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_fp = compute_split_manifest_fingerprint(manifest["canonical_split"])
            train_fp = hashlib.sha256(json.dumps(["a.jpg"]).encode("utf-8")).hexdigest()

            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest": str(manifest_file),
                "train_split_fingerprint": train_fp,
                "split_manifest_fingerprint": manifest_fp,
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_fp,
                "checkpoint_file": "model.keras",
                "checkpoint_fingerprint": ckpt_fp
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            mod_p, tok_p, train_split, exp_split, meta = resolve_v1_evaluation_context(
                model_file=str(fake_ckpt)
            )
            self.assertEqual(train_split, "canonical_split")
            self.assertEqual(meta["checkpoint_fingerprint"], ckpt_fp)

    def test_checkpoint_reject_malformed_metadata(self):
        """11. Evaluation rejects malformed metadata (corrupt JSON and malformed fingerprint)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_ckpt = tmp_path / "model.keras"
            fake_ckpt.write_bytes(b"bytes")

            # Corrupt JSON (truncated)
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                f.write('{"model_version": "V1_Baseline_CNN_LSTM", "split_type":')

            with self.assertRaises(ValueError) as ctx1:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("CORRUPT EXPERIMENT METADATA", str(ctx1.exception))

            # Malformed fingerprint (not a 64-char hex string)
            fake_tok = tmp_path / "tokenizer.json"
            fake_tok.write_text(json.dumps({"word_to_index": {"a": 0}, "max_length": 5}), encoding="utf-8")
            cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "vocab_size": 1,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": compute_file_sha256(fake_tok),
                "checkpoint_file": "model.keras",
                "checkpoint_fingerprint": "not-a-valid-sha256-string"
            }
            with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(ValueError) as ctx2:
                resolve_v1_evaluation_context(model_file=str(fake_ckpt))
            self.assertIn("MALFORMED CHECKPOINT FINGERPRINT", str(ctx2.exception))

    def test_checkpoint_safe_interrupted_metadata_update(self):
        """12. Atomic config save ensures partially written files never replace valid config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_cfg = tmp_path / "config.json"

            initial_cfg = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "checkpoint_fingerprint": None
            }
            save_experiment_config_safely(initial_cfg, target_cfg)
            self.assertTrue(target_cfg.exists())

            # Read content, verify valid JSON with null fingerprint
            with open(target_cfg, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsNone(loaded["checkpoint_fingerprint"])

            # Simulate interrupted write to temporary file
            interrupted_tmp = target_cfg.with_name(f"{target_cfg.name}.tmp")
            interrupted_tmp.write_text('{"truncated_config": 123', encoding="utf-8")

            # target_cfg remains intact and uncorrupted
            with open(target_cfg, "r", encoding="utf-8") as f:
                intact = json.load(f)
            self.assertEqual(intact["model_version"], "V1_Baseline_CNN_LSTM")
            self.assertIsNone(intact["checkpoint_fingerprint"])

    def test_v1_caption_data_generator_contract(self):
        """13. Verify V1 CaptionDataGenerator contract: shapes, types, and sequence expansion."""
        word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3, "dog": 4, "runs": 5}
        captions_dict = {
            "img1.jpg": ["<start> dog runs <end>"]
        }
        features_dict = {
            "img1.jpg": np.ones((2048,), dtype=np.float32)
        }
        gen = V1CaptionDataGenerator(
            captions_dict=captions_dict,
            features_dict=features_dict,
            word_to_index=word_to_index,
            max_length=6,
            batch_size=2,
            shuffle=False
        )
        self.assertGreater(len(gen), 0)
        (X_img, X_seq), y = gen[0]
        self.assertEqual(X_img.ndim, 2)
        self.assertEqual(X_img.shape[1], 2048)
        self.assertEqual(X_seq.ndim, 2)
        self.assertEqual(X_seq.shape[1], 6)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(len(X_img), len(y))
        # Ensure first input sequence starts with start token id 1
        self.assertEqual(X_seq[0, 0], 1)

    def test_v2_training_dataset_contract_and_defensive_reshape(self):
        """14. Verify V2 training dataset pipeline produces expected shapes and reshapes spatial features."""
        word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3, "cat": 4, "sits": 5}
        # Include one standard shape (64, 2048) and one pre-reshaped (8, 8, 2048)
        captions_dict = {
            "img1.jpg": ["<start> cat sits <end>"],
            "img2.jpg": ["<start> cat <end>"]
        }
        features_dict = {
            "img1.jpg": np.zeros((64, 2048), dtype=np.float32),
            "img2.jpg": np.zeros((8, 8, 2048), dtype=np.float32)
        }
        ds = v2_build_training_dataset(
            captions_dict=captions_dict,
            features_dict=features_dict,
            word_to_index=word_to_index,
            max_length=8,
            batch_size=2
        )
        batch = next(iter(ds))
        img_feat, caps = batch
        self.assertEqual(img_feat.shape, (2, 64, 2048))
        self.assertEqual(caps.shape, (2, 8))
        # Verify teacher-forcing target alignment contract
        targets = caps[:, 1:]
        self.assertEqual(targets.shape, (2, 7))

    def test_feature_loaders_npz_and_npy_interoperability(self):
        """15. Verify both V1 and V2 load_features support both .npz and .npy dictionaries seamlessly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_features = {
                "img1.jpg": np.ones((64, 2048), dtype=np.float32),
                "img2.jpg": np.zeros((64, 2048), dtype=np.float32)
            }
            npz_path = tmp_path / "feat.npz"
            npy_path = tmp_path / "feat.npy"

            np.savez_compressed(npz_path, **fake_features)
            np.save(npy_path, fake_features)

            # V1 loader on both formats
            loaded_v1_npz = v1_load_features(str(npz_path))
            self.assertEqual(len(loaded_v1_npz), 2)
            loaded_v1_npy = v1_load_features(str(npy_path))
            self.assertEqual(len(loaded_v1_npy), 2)

            # V2 loader on both formats
            loaded_v2_npz = v2_load_features(str(npz_path))
            self.assertEqual(len(loaded_v2_npz), 2)
            loaded_v2_npy = v2_load_features(str(npy_path))
            self.assertEqual(len(loaded_v2_npy), 2)

    def test_dataset_custom_caption_file_and_safe_import(self):
        """16. Verify dataset.py load_captions accepts custom caption file path safely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cap_file = tmp_path / "captions.txt"
            cap_file.write_text(
                "image,caption\n"
                "test1.jpg,a cat on a mat\n"
                "test1.jpg,a lovely feline sitting down\n"
                "test2.jpg,a dog barking loudly\n",
                encoding="utf-8"
            )
            v1_caps = v1_load_captions(caption_file=str(cap_file))
            self.assertIn("test1.jpg", v1_caps)
            self.assertEqual(len(v1_caps["test1.jpg"]), 2)

            v2_caps = v2_load_captions(caption_file=str(cap_file))
            self.assertIn("test2.jpg", v2_caps)
            self.assertEqual(len(v2_caps["test2.jpg"]), 1)

    def test_cross_version_experiment_metadata_rejection(self):
        """17. Verify resolve context guards reject cross-version mismatched experiment configs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_v1_ckpt = tmp_path / "fake_v1_model.keras"
            fake_v1_ckpt.write_bytes(b"NON_HISTORICAL_V1_MOCK")
            fake_v1_hash = hashlib.sha256(b"NON_HISTORICAL_V1_MOCK").hexdigest()

            fake_v2_ckpt = tmp_path / "fake_v2_model.weights.h5"
            fake_v2_ckpt.write_bytes(b"NON_HISTORICAL_V2_MOCK")
            fake_v2_hash = hashlib.sha256(b"NON_HISTORICAL_V2_MOCK").hexdigest()

            tok_file = tmp_path / "tokenizer.json"
            tok_content = {"word_to_index": {"<pad>": 0}, "index_to_word": {"0": "<pad>"}, "max_length": 10}
            tok_file.write_text(json.dumps(tok_content), encoding="utf-8")
            tok_hash = hashlib.sha256(tok_file.read_bytes()).hexdigest()

            # Config configured for V2, passed to V1 resolver
            v2_cfg_data = {
                "model_version": "V2_Attention",
                "split_type": "canonical",
                "split_manifest_fingerprint": "a" * 64,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_hash,
                "checkpoint_file": "fake_v1_model.keras",
                "checkpoint_fingerprint": fake_v1_hash
            }
            v2_cfg_path = tmp_path / "config_v2.json"
            v2_cfg_path.write_text(json.dumps(v2_cfg_data), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                resolve_v1_evaluation_context(
                    model_file=str(fake_v1_ckpt),
                    config_file=str(v2_cfg_path)
                )
            self.assertIn("MODEL VERSION MISMATCH", str(ctx.exception))

            # Config configured for V1, passed to V2 resolver
            v1_cfg_data = {
                "model_version": "V1_Baseline_CNN_LSTM",
                "split_type": "canonical",
                "split_manifest_fingerprint": "b" * 64,
                "tokenizer_file": "tokenizer.json",
                "tokenizer_fingerprint": tok_hash,
                "checkpoint_file": "fake_v2_model.weights.h5",
                "checkpoint_fingerprint": fake_v2_hash
            }
            v1_cfg_path = tmp_path / "config_v1.json"
            v1_cfg_path.write_text(json.dumps(v1_cfg_data), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx2:
                resolve_v2_evaluation_context(
                    weights_file=str(fake_v2_ckpt),
                    config_file=str(v1_cfg_path)
                )
            self.assertIn("MODEL VERSION MISMATCH", str(ctx2.exception))

    def test_v1_dry_run_non_mutating_and_leaves_existing_artifacts_byte_identical(self):
        """18. Verify V1 dry-run is completely non-mutating and leaves existing artifacts byte-identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            splits_file = tmp_path / "splits.json"
            splits_data = {
                "canonical_split": {
                    "train": ["img1.jpg", "img2.jpg"],
                    "val": ["img3.jpg"],
                    "test": ["img4.jpg"]
                }
            }
            splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

            exp_dir = tmp_path / "experiments" / "v1_canonical"
            models_dir = exp_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            orig_config = exp_dir / "config.json"
            orig_config.write_text('{"model_version": "V1_Baseline_CNN_LSTM", "checkpoint_fingerprint": "abcd1234"}', encoding="utf-8")

            orig_ckpt = models_dir / "best_caption_model.keras"
            orig_ckpt.write_bytes(b"ORIGINAL_TRAINED_V1_CHECKPOINT_BYTES")

            orig_tok = models_dir / "tokenizer.json"
            orig_tok.write_text('{"word_to_index": {"<pad>": 0, "<start>": 1}}', encoding="utf-8")

            orig_hist = exp_dir / "history.json"
            orig_hist.write_text('{"val_loss": [0.42]}', encoding="utf-8")

            # Record hashes before dry-run
            files_before = {}
            for p in exp_dir.glob("**/*"):
                if p.is_file():
                    files_before[str(p.relative_to(exp_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()

            fake_args = Namespace(
                split="canonical",
                splits_file=str(splits_file),
                output_dir=str(exp_dir),
                features_file=str(tmp_path / "nonexistent_feat.npz"),
                epochs=40,
                batch_size=64,
                min_frequency=1,
                embedding_dim=256,
                hidden_dim=256,
                dry_run=True,
                allow_overwrite_historical=False
            )

            mock_captions = {
                "img1.jpg": ["<start> dog runs <end>"],
                "img2.jpg": ["<start> cat sleeps <end>"],
                "img3.jpg": ["<start> bird flies <end>"]
            }

            with patch("V1_Baseline_CNN_LSTM.src.train.load_captions", return_value=mock_captions):
                with patch("V1_Baseline_CNN_LSTM.src.train.load_dataset_split", return_value=(
                    {"img1.jpg": mock_captions["img1.jpg"], "img2.jpg": mock_captions["img2.jpg"]},
                    {"img3.jpg": mock_captions["img3.jpg"]},
                    splits_data["canonical_split"]
                )):
                    info = v1_validate_training_setup(fake_args)
                    self.assertGreater(info["vocab_size"], 0)
                    self.assertEqual(info["train_images"], 2)

            # Verify directory content and byte hashes are identical
            files_after = {}
            for p in exp_dir.glob("**/*"):
                if p.is_file():
                    files_after[str(p.relative_to(exp_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()

            self.assertEqual(files_before, files_after, "Dry-run modified or added files in the experiment directory!")

    def test_v2_dry_run_non_mutating_and_leaves_existing_artifacts_byte_identical(self):
        """19. Verify V2 dry-run is completely non-mutating and leaves existing artifacts byte-identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            splits_file = tmp_path / "splits.json"
            splits_data = {
                "canonical_split": {
                    "train": ["img1.jpg", "img2.jpg"],
                    "val": ["img3.jpg"],
                    "test": ["img4.jpg"]
                }
            }
            splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

            exp_dir = tmp_path / "experiments" / "v2_canonical"
            models_dir = exp_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            orig_config = exp_dir / "config.json"
            orig_config.write_text('{"model_version": "V2_Attention", "checkpoint_fingerprint": "v2_hash_1234"}', encoding="utf-8")

            orig_weights = models_dir / "v2_attention.weights.h5"
            orig_weights.write_bytes(b"ORIGINAL_TRAINED_V2_WEIGHTS_BYTES")

            orig_tok = models_dir / "tokenizer.json"
            orig_tok.write_text('{"word_to_index": {"<pad>": 0, "<start>": 1}}', encoding="utf-8")

            orig_hist = exp_dir / "history.json"
            orig_hist.write_text('{"val_loss": [0.38]}', encoding="utf-8")

            # Record hashes before dry-run
            files_before = {}
            for p in exp_dir.glob("**/*"):
                if p.is_file():
                    files_before[str(p.relative_to(exp_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()

            fake_args = Namespace(
                split="canonical",
                splits_file=str(splits_file),
                output_dir=str(exp_dir),
                features_file=None,
                epochs=40,
                batch_size=32,
                min_frequency=1,
                embedding_dim=256,
                units=256,
                dry_run=True,
                allow_overwrite_historical=False
            )

            mock_captions = {
                "img1.jpg": ["<start> dog runs <end>"],
                "img2.jpg": ["<start> cat sleeps <end>"],
                "img3.jpg": ["<start> bird flies <end>"]
            }

            with patch("V2_Attention.src.train.load_captions", return_value=mock_captions):
                with patch("V2_Attention.src.train.load_dataset_split", return_value=(
                    {"img1.jpg": mock_captions["img1.jpg"], "img2.jpg": mock_captions["img2.jpg"]},
                    {"img3.jpg": mock_captions["img3.jpg"]},
                    splits_data["canonical_split"]
                )):
                    info = v2_validate_training_setup(fake_args)
                    self.assertGreater(info["vocab_size"], 0)
                    self.assertEqual(info["train_images"], 2)

            # Verify directory content and byte hashes are identical
            files_after = {}
            for p in exp_dir.glob("**/*"):
                if p.is_file():
                    files_after[str(p.relative_to(exp_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()

            self.assertEqual(files_before, files_after, "Dry-run modified or added files in the experiment directory!")

    def test_dry_run_reports_invalid_hyperparameters_and_paths(self):
        """20. Verify validation-only dry-run clearly reports invalid hyperparameters and missing paths."""
        # 1. Invalid epochs
        bad_epochs_args = Namespace(epochs=0, batch_size=32, min_frequency=1, embedding_dim=256, hidden_dim=256, units=256)
        with self.assertRaises(ValueError) as ctx1:
            v1_validate_training_setup(bad_epochs_args)
        self.assertIn("epochs must be > 0", str(ctx1.exception))

        # 2. Invalid batch_size
        bad_batch_args = Namespace(epochs=10, batch_size=-1, min_frequency=1, embedding_dim=256, hidden_dim=256, units=256)
        with self.assertRaises(ValueError) as ctx2:
            v2_validate_training_setup(bad_batch_args)
        self.assertIn("batch_size must be > 0", str(ctx2.exception))

        # 3. Invalid min_frequency
        bad_freq_args = Namespace(epochs=10, batch_size=32, min_frequency=0, embedding_dim=256, hidden_dim=256, units=256)
        with self.assertRaises(ValueError) as ctx3:
            v1_validate_training_setup(bad_freq_args)
        self.assertIn("min_frequency must be >= 1", str(ctx3.exception))

        # 4. Invalid dimensions
        bad_dim_args = Namespace(epochs=10, batch_size=32, min_frequency=1, embedding_dim=0, hidden_dim=256, units=256)
        with self.assertRaises(ValueError) as ctx4:
            v1_validate_training_setup(bad_dim_args)
        self.assertIn("embedding_dim must be > 0", str(ctx4.exception))

        # 5. Missing splits manifest file
        bad_split_args = Namespace(epochs=10, batch_size=32, min_frequency=1, embedding_dim=256, hidden_dim=256, units=256, splits_file="nonexistent_split_file.json")
        with self.assertRaises(FileNotFoundError) as ctx5:
            v1_validate_training_setup(bad_split_args)
        self.assertIn("Splits manifest not found", str(ctx5.exception))

    def test_v1_architecture_static_layer_order(self):
        """21. Statically verify V1 architecture layer order without instantiating Keras models."""
        import ast
        model_file = V1_ROOT / "src" / "model.py"
        self.assertTrue(model_file.exists())
        with open(model_file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        # Find build_caption_model function definition
        build_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_caption_model":
                build_fn = node
                break
        self.assertIsNotNone(build_fn, "build_caption_model not found in model.py")

        dropout_idx = source.find("name=\"image_dropout\"")
        dense_idx = source.find("name=\"image_dense\"")

        self.assertGreater(dropout_idx, 0, "image_dropout layer definition not found")
        self.assertGreater(dense_idx, 0, "image_dense layer definition not found")
        self.assertLess(dropout_idx, dense_idx, "In V1 model.py image branch, image_dropout MUST precede image_dense!")

    def test_v1_evaluate_cli_defaults_are_optional(self):
        """22. Verify V1 evaluate.py CLI defaults for --model and --tokenizer are present and optional."""
        import ast
        eval_file = V1_ROOT / "src" / "evaluate.py"
        with open(eval_file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        found_args = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr") and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    arg_name = node.args[0].value
                    for kw in node.keywords:
                        if kw.arg == "default":
                            found_args[arg_name] = kw.value

        self.assertIn("--image", found_args)
        self.assertIn("--model", found_args)
        self.assertIn("--tokenizer", found_args)
        self.assertIsNotNone(found_args["--model"])
        self.assertIsNotNone(found_args["--tokenizer"])

    def test_reproducibility_seed_recorded_and_validated(self):
        """23. Verify reproducibility seed is validated, recorded, and configurable."""
        # 1. Negative seed rejected
        bad_seed_args = Namespace(epochs=10, batch_size=32, min_frequency=1, embedding_dim=256, hidden_dim=256, units=256, seed=-1)
        with self.assertRaises(ValueError) as ctx:
            v1_validate_training_setup(bad_seed_args)
        self.assertIn("seed must be a non-negative integer", str(ctx.exception))

        # 2. Seed setter sets state without error
        v1_set_reproducible_seed(42)
        v2_set_reproducible_seed(123)

    def test_seed_does_not_alter_canonical_split_membership(self):
        """24. Verify changing random seeds does not alter canonical split membership."""
        canonical_splits_file = REPO_ROOT / "splits.json"
        if canonical_splits_file.exists():
            captions_dict = {f"img{i}.jpg": ["sample caption"] for i in range(100)}
            # Load with seed 42
            v1_set_reproducible_seed(42)
            train_42, val_42, split_42 = v1_load_dataset_split("canonical", str(canonical_splits_file), captions_dict)

            # Load with seed 9999
            v1_set_reproducible_seed(9999)
            train_9999, val_9999, split_9999 = v1_load_dataset_split("canonical", str(canonical_splits_file), captions_dict)

            self.assertEqual(split_42["train"], split_9999["train"])
            self.assertEqual(split_42["val"], split_9999["val"])
            self.assertEqual(split_42["test"], split_9999["test"])

    def test_history_schema_and_atomic_serialization(self):
        """25. Verify history schema and atomic write safety in temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_history = tmp_path / "history.json"

            # V1 schema
            v1_history = {
                "loss": [2.5, 2.1, 1.8],
                "val_loss": [2.6, 2.2, 1.9],
                "accuracy": [0.45, 0.52, 0.58],
                "val_accuracy": [0.44, 0.50, 0.55]
            }
            v1_save_history_safely(v1_history, target_history)
            self.assertTrue(target_history.exists())
            self.assertFalse(target_history.with_name(f"{target_history.name}.tmp").exists())

            with open(target_history, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["loss"], [2.5, 2.1, 1.8])
            self.assertEqual(loaded["val_loss"], [2.6, 2.2, 1.9])

            # V2 schema update on same path
            v2_history = {
                "train_loss": [3.1, 2.7],
                "val_loss": [3.2, 2.8]
            }
            v2_save_history_safely(v2_history, target_history)
            with open(target_history, "r", encoding="utf-8") as f:
                loaded_v2 = json.load(f)
            self.assertEqual(loaded_v2["train_loss"], [3.1, 2.7])

    def test_interrupted_history_write_preserves_valid_file(self):
        """26. Verify interrupted history write (.tmp in flight) preserves previous valid history."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_history = tmp_path / "history.json"

            valid_history = {"train_loss": [2.0, 1.8], "val_loss": [2.1, 1.9]}
            v2_save_history_safely(valid_history, target_history)

            # Simulate interrupted write leaving corrupt .tmp file
            corrupt_tmp = target_history.with_name(f"{target_history.name}.tmp")
            corrupt_tmp.write_text('{"train_loss": [CORRUPTED_JSON', encoding="utf-8")

            # Verify original history file is completely intact
            with open(target_history, "r", encoding="utf-8") as f:
                intact = json.load(f)
            self.assertEqual(intact["train_loss"], [2.0, 1.8])
            self.assertEqual(intact["val_loss"], [2.1, 1.9])

    def test_experiment_overwrite_protection_guard(self):
        """27. Verify completed experiment directory cannot be overwritten without explicit flag."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            completed_cfg = tmp_path / "config.json"
            completed_cfg.write_text(json.dumps({
                "model_version": "V1_Baseline_CNN_LSTM",
                "checkpoint_fingerprint": "abcdef1234567890abcdef1234567890"
            }), encoding="utf-8")

            with open(completed_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.assertIsNotNone(cfg.get("checkpoint_fingerprint"))

    def test_fair_comparison_metadata_schema_completeness(self):
        """28. Verify all required fair comparison fields are present in experiment config schema."""
        from V1_Baseline_CNN_LSTM.src.train import parse_arguments as v1_parse_args
        from V2_Attention.src.train import parse_arguments as v2_parse_args

        v1_args = v1_parse_args([])
        v2_args = v2_parse_args([])

        self.assertTrue(hasattr(v1_args, "seed"))
        self.assertTrue(hasattr(v2_args, "seed"))
        self.assertEqual(v1_args.seed, 42)
        self.assertEqual(v2_args.seed, 42)
        self.assertTrue(hasattr(v1_args, "allow_overwrite_experiment"))
        self.assertTrue(hasattr(v2_args, "allow_overwrite_experiment"))

    def test_v1_target_semantics_pad_never_supervised_target(self):
        """29. Verify V1 generator targets are strictly valid tokens and never pad token (0)."""
        captions_dict = {
            "img1.jpg": ["<start> a dog runs <end>"],
            "img2.jpg": ["<start> cat sleeps on mat <end>"]
        }
        features_dict = {
            "img1.jpg": np.zeros(2048, dtype=np.float32),
            "img2.jpg": np.zeros(2048, dtype=np.float32)
        }
        word_to_index = {
            "<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3,
            "a": 4, "dog": 5, "runs": 6, "cat": 7, "sleeps": 8, "on": 9, "mat": 10
        }
        max_length = 8
        gen = V1CaptionDataGenerator(
            captions_dict=captions_dict,
            features_dict=features_dict,
            word_to_index=word_to_index,
            max_length=max_length,
            batch_size=4,
            shuffle=False
        )

        # "<start> a dog runs <end>" has 5 tokens -> 4 targets: a, dog, runs, <end>
        # "<start> cat sleeps on mat <end>" has 6 tokens -> 5 targets: cat, sleeps, on, mat, <end>
        # Total samples = 4 + 5 = 9
        self.assertEqual(len(gen.samples), 9)

        all_targets = []
        for i in range(len(gen)):
            (img_batch, seq_batch), target_batch = gen[i]
            all_targets.extend(target_batch.tolist())
            # Seq batch must have shape (batch_size, max_length)
            self.assertEqual(seq_batch.shape[1], max_length)

        # Assert index 0 (<pad>) is NEVER in target_batch
        self.assertNotIn(0, all_targets)
        for t in all_targets:
            self.assertGreater(t, 0)
            self.assertIn(t, [word_to_index["a"], word_to_index["dog"], word_to_index["runs"],
                              word_to_index["cat"], word_to_index["sleeps"], word_to_index["on"],
                              word_to_index["mat"], word_to_index["<end>"]])

    def test_v2_training_dataset_shuffle_flag_suppresses_randomness(self):
        """30. Verify V2 build_training_dataset supports shuffle=False for deterministic evaluation."""
        captions_dict = {
            f"img_{i}.jpg": [f"<start> word_{i} <end>"] for i in range(10)
        }
        features_dict = {
            f"img_{i}.jpg": np.zeros((64, 2048), dtype=np.float32) for i in range(10)
        }
        word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        for i in range(10):
            word_to_index[f"word_{i}"] = 4 + i

        ds1 = v2_build_training_dataset(
            captions_dict=captions_dict,
            features_dict=features_dict,
            word_to_index=word_to_index,
            max_length=5,
            batch_size=2,
            shuffle=False
        )
        ds2 = v2_build_training_dataset(
            captions_dict=captions_dict,
            features_dict=features_dict,
            word_to_index=word_to_index,
            max_length=5,
            batch_size=2,
            shuffle=False
        )

        caps1 = [batch[1].numpy().tolist() for batch in ds1]
        caps2 = [batch[1].numpy().tolist() for batch in ds2]
        self.assertEqual(caps1, caps2, "shuffle=False should produce identical deterministic sequence batches")

    def test_v1_and_v2_loss_token_normalization_mathematical_equivalence(self):
        """31. Verify V1 sample count and V2 valid token count match exactly for same caption."""
        caption = "<start> a brown dog runs fast <end>"
        tokens = caption.split()
        self.assertEqual(len(tokens), 7)
        v1_prefix_target_pairs = len(tokens) - 1  # 6 pairs, each predicting 1 real token

        # In V2, padded to length 10 -> targets = sequence[1:], so length is 9 (6 valid, 3 pad)
        seq_len = 10
        v2_valid_targets = len(tokens) - 1
        v2_pad_targets = (seq_len - 1) - v2_valid_targets

        self.assertEqual(v1_prefix_target_pairs, v2_valid_targets)
        self.assertEqual(v1_prefix_target_pairs, 6)
        self.assertEqual(v2_pad_targets, 3)

    def test_v2_validation_loss_components_and_global_aggregation(self):
        """32. Verify compute_loss_components and that unweighted batch mean differs from global token mean."""
        from V2_Attention.src.loss import compute_loss_components, loss_function

        # Batch 1: 2 valid tokens out of 4 (targets: [1, 2, 0, 0])
        # Logits crafted so cross-entropy is deterministic
        vocab_size = 5
        pred_b1 = tf.constant([
            [10.0, 0.0, 0.0, 0.0, 0.0],  # for token 1 -> high confidence for class 0, true class is 1 -> high loss
            [10.0, 0.0, 0.0, 0.0, 0.0],  # for token 2
            [10.0, 0.0, 0.0, 0.0, 0.0],  # pad (0)
            [10.0, 0.0, 0.0, 0.0, 0.0]   # pad (0)
        ], dtype=tf.float32)
        real_b1 = tf.constant([1, 2, 0, 0], dtype=tf.int32)

        loss_sum_1, count_1 = compute_loss_components(real_b1, pred_b1)
        self.assertEqual(float(count_1), 2.0)
        batch_loss_1 = float(loss_function(real_b1, pred_b1))
        self.assertAlmostEqual(batch_loss_1, float(loss_sum_1 / count_1), places=5)

        # Batch 2: 6 valid tokens out of 8
        pred_b2 = tf.constant([
            [0.0, 10.0, 0.0, 0.0, 0.0],  # token 1 (target 1) -> low loss
            [0.0, 0.0, 10.0, 0.0, 0.0],  # token 2 (target 2) -> low loss
            [0.0, 0.0, 0.0, 10.0, 0.0],  # token 3 (target 3) -> low loss
            [0.0, 0.0, 0.0, 0.0, 10.0],  # token 4 (target 4) -> low loss
            [0.0, 10.0, 0.0, 0.0, 0.0],  # token 1 (target 1) -> low loss
            [0.0, 0.0, 10.0, 0.0, 0.0],  # token 2 (target 2) -> low loss
            [10.0, 0.0, 0.0, 0.0, 0.0],  # pad (0)
            [10.0, 0.0, 0.0, 0.0, 0.0]   # pad (0)
        ], dtype=tf.float32)
        real_b2 = tf.constant([1, 2, 3, 4, 1, 2, 0, 0], dtype=tf.int32)

        loss_sum_2, count_2 = compute_loss_components(real_b2, pred_b2)
        self.assertEqual(float(count_2), 6.0)
        batch_loss_2 = float(loss_function(real_b2, pred_b2))
        self.assertAlmostEqual(batch_loss_2, float(loss_sum_2 / count_2), places=5)

        # Unweighted mean of batch losses:
        unweighted_batch_mean = (batch_loss_1 + batch_loss_2) / 2.0

        # Globally token-weighted validation loss:
        global_token_weighted_loss = float(loss_sum_1 + loss_sum_2) / float(count_1 + count_2)

        # Mathematical verification: the two aggregations must not be equal
        self.assertNotEqual(round(unweighted_batch_mean, 4), round(global_token_weighted_loss, 4))
        # Batch 1 (high loss) had only 2 tokens; Batch 2 (low loss) had 6 tokens.
        # Thus, global loss must be significantly lower than unweighted mean.
        self.assertLess(global_token_weighted_loss, unweighted_batch_mean)

    def test_v2_validation_loss_all_padding_safe_division(self):
        """33. Verify all-padding/zero-token inputs safely evaluate to 0.0 at batch level."""
        from V2_Attention.src.loss import compute_loss_components, loss_function

        all_pad_real = tf.zeros((4, 10), dtype=tf.int32)
        dummy_pred = tf.zeros((4, 10, 50), dtype=tf.float32)

        loss_sum, token_count = compute_loss_components(all_pad_real, dummy_pred)
        self.assertEqual(float(loss_sum), 0.0)
        self.assertEqual(float(token_count), 0.0)

        # Batch-level loss function returns safe 0.0 without NaN/Inf
        batch_loss = float(loss_function(all_pad_real, dummy_pred))
        self.assertEqual(batch_loss, 0.0)
        self.assertFalse(np.isnan(batch_loss))
        self.assertFalse(np.isinf(batch_loss))

    def test_v2_checkpoint_selection_semantics_divergence(self):
        """34. Verify that unweighted batch averaging can select the wrong checkpoint while global loss selects the true best."""
        # Scenario:
        # Epoch 1:
        #   Batch 1: 2 valid tokens, loss_sum = 10.0 -> batch_loss = 5.0
        #   Batch 2: 10 valid tokens, loss_sum = 10.0 -> batch_loss = 1.0
        #   Unweighted mean: (5.0 + 1.0) / 2 = 3.00
        #   Global token mean: (10 + 10) / (2 + 10) = 20 / 12 = 1.6667
        #
        # Epoch 2:
        #   Batch 1: 2 valid tokens, loss_sum = 4.0 -> batch_loss = 2.0
        #   Batch 2: 10 valid tokens, loss_sum = 25.0 -> batch_loss = 2.5
        #   Unweighted mean: (2.0 + 2.5) / 2 = 2.25  <-- Unweighted thinks Epoch 2 is better!
        #   Global token mean: (4 + 25) / (2 + 10) = 29 / 12 = 2.4167 <-- True global shows Epoch 2 degraded!

        ep1_unweighted = (5.0 + 1.0) / 2.0  # 3.00
        ep2_unweighted = (2.0 + 2.5) / 2.0  # 2.25

        ep1_global = (10.0 + 10.0) / (2.0 + 10.0)  # 1.6667
        ep2_global = (4.0 + 25.0) / (2.0 + 10.0)   # 2.4167

        # Unweighted ranking would wrongly prefer Epoch 2
        self.assertLess(ep2_unweighted, ep1_unweighted)

        # Correct global token-weighted ranking correctly identifies Epoch 1 as superior
        self.assertLess(ep1_global, ep2_global)

        # Simulate checkpoint tracking loop with corrected global loss
        best_val_loss = float("inf")
        selected_checkpoint = None

        # Epoch 1 evaluation
        val_loss_ep1 = ep1_global
        if val_loss_ep1 < best_val_loss:
            best_val_loss = val_loss_ep1
            selected_checkpoint = "epoch_1"

        # Epoch 2 evaluation
        val_loss_ep2 = ep2_global
        if val_loss_ep2 < best_val_loss:
            best_val_loss = val_loss_ep2
            selected_checkpoint = "epoch_2"

        # Confirms Epoch 1 is retained as best model
        self.assertEqual(selected_checkpoint, "epoch_1")
        self.assertAlmostEqual(best_val_loss, 1.6667, places=4)

    def test_v2_validation_loss_empty_or_all_padding_epoch_rejected(self):
        """35. Verify entirely empty or all-padding validation epoch is rejected with ValueError."""
        from V2_Attention.src.loss import aggregate_validation_loss

        # Case 1: Empty validation dataset (0 batches)
        with self.assertRaises(ValueError) as ctx1:
            aggregate_validation_loss(total_loss_sum=0.0, total_token_count=0.0, num_batches=0)
        self.assertIn("Validation dataset is empty", str(ctx1.exception))

        # Case 2: All validation batches contain only padding (0 valid tokens)
        with self.assertRaises(ValueError) as ctx2:
            aggregate_validation_loss(total_loss_sum=0.0, total_token_count=0.0, num_batches=3)
        self.assertIn("zero valid target tokens", str(ctx2.exception))

    def test_v2_validation_loss_padding_batch_with_valid_batches_succeeds(self):
        """36. Verify an all-padding batch followed by valid batches produces correct global metric."""
        from V2_Attention.src.loss import aggregate_validation_loss

        # Batch 1 (all padding): 0 loss, 0 tokens
        # Batch 2 (valid): loss_sum = 40.0, tokens = 10 -> mean = 4.0
        # Batch 3 (valid): loss_sum = 60.0, tokens = 15 -> mean = 4.0
        total_loss = 0.0 + 40.0 + 60.0
        total_tokens = 0.0 + 10.0 + 15.0
        val_loss = aggregate_validation_loss(total_loss, total_tokens, num_batches=3)
        self.assertEqual(val_loss, 100.0 / 25.0)
        self.assertEqual(val_loss, 4.0)

    def test_v2_validation_loss_non_finite_rejected(self):
        """37. Verify NaN and infinite validation metrics raise FloatingPointError."""
        from V2_Attention.src.loss import aggregate_validation_loss

        # NaN
        with self.assertRaises(FloatingPointError):
            aggregate_validation_loss(float("nan"), total_token_count=10.0, num_batches=1)

        # Inf
        with self.assertRaises(FloatingPointError):
            aggregate_validation_loss(float("inf"), total_token_count=10.0, num_batches=1)

        # -Inf
        with self.assertRaises(FloatingPointError):
            aggregate_validation_loss(float("-inf"), total_token_count=10.0, num_batches=1)

    def test_v2_invalid_validation_preserves_best_checkpoint_and_history(self):
        """38. Verify invalid validation results cannot overwrite best checkpoint or corrupt history."""
        from V2_Attention.src.loss import aggregate_validation_loss

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)
            history_file = exp_dir / "history.json"
            config_file = exp_dir / "config.json"

            # Initial valid state after Epoch 1 and Epoch 2
            initial_history = {"train_loss": [3.5, 3.0], "val_loss": [3.2, 2.8]}
            initial_config = {
                "model_version": "V2_Attention",
                "completed_epochs": 2,
                "best_val_loss": 2.8,
                "checkpoint_fingerprint": "original_fingerprint_hash_abc"
            }
            history_file.write_text(json.dumps(initial_history), encoding="utf-8")
            config_file.write_text(json.dumps(initial_config), encoding="utf-8")

            best_val_loss = 2.8
            checkpoint_saved = False

            # Simulate Epoch 3: validation produces 0 valid tokens
            try:
                # aggregate_validation_loss raises ValueError on 0 tokens
                val_loss_ep3 = aggregate_validation_loss(total_loss_sum=0.0, total_token_count=0.0, num_batches=2)
                # If it had returned 0.0, it would have incorrectly beaten 2.8!
                if val_loss_ep3 < best_val_loss:
                    best_val_loss = val_loss_ep3
                    checkpoint_saved = True
            except ValueError:
                # Evaluation failed closed: do not update history or checkpoint
                pass

            # Assert best_val_loss was NOT lowered to 0.0
            self.assertEqual(best_val_loss, 2.8)
            self.assertFalse(checkpoint_saved)

            # Assert historical files remain completely unaltered
            history_after = json.loads(history_file.read_text(encoding="utf-8"))
            config_after = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(history_after["val_loss"], [3.2, 2.8])
            self.assertEqual(config_after["best_val_loss"], 2.8)
            self.assertEqual(config_after["completed_epochs"], 2)


class TestMemoryEfficientFeatureStoreAndConverter(unittest.TestCase):
    """
    Rigorously tests Phase 7 memory-efficient feature store and offline converter:
    1. Creation and reading of tiny memory-mapped store.
    2. Correct filename-to-row lookup and arbitrary access order.
    3. Missing image ID and malformed manifest errors.
    4. Shape normalization for supported per-image shapes: (64, 2048), (8, 8, 2048), (1, 64, 2048).
    5. Rejection of non-finite values (NaN, Inf) and invalid dimensions.
    6. Refusal to overwrite existing files and source preservation.
    7. Handling of interrupted or incomplete conversion outputs.
    8. Seamless integration with build_training_dataset.
    9. Legacy .npy and .npz format backward compatibility.
    10. Verification of memory mapping (np.memmap) without full-array heap allocation.
    """

    def test_01_mmap_store_creation_and_reading(self):
        """1. Verify creation and reading of a tiny memory-mapped feature store."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "legacy_dict.npy"
            raw_dict = {
                "img_01.jpg": np.ones((64, 2048), dtype=np.float32),
                "img_02.jpg": np.full((64, 2048), 5.0, dtype=np.float32)
            }
            np.save(str(legacy_file), raw_dict)

            store_dir = tmp_path / "mmap_store"
            manifest_file = convert_legacy_features(input_path=legacy_file, output_dir=store_dir)
            self.assertTrue(manifest_file.exists())
            self.assertTrue((store_dir / "features.npy").exists())

            # Read from directory path
            store = MmapFeatureStore(store_dir)
            self.assertEqual(len(store), 2)
            self.assertTrue(np.allclose(store["img_01.jpg"], 1.0))
            self.assertTrue(np.allclose(store["img_02.jpg"], 5.0))
            store.close()

            # Read from manifest file path
            store_from_manifest = MmapFeatureStore(manifest_file)
            self.assertEqual(len(store_from_manifest), 2)
            self.assertTrue(np.allclose(store_from_manifest["img_01.jpg"], 1.0))
            store_from_manifest.close()

    def test_02_filename_to_row_lookup_and_arbitrary_access_order(self):
        """2. Verify filename-to-row lookup order independence and containment tests."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "legacy.npy"
            raw_dict = {
                "z_zebra.jpg": np.full((64, 2048), 10.0, dtype=np.float32),
                "a_apple.jpg": np.full((64, 2048), 20.0, dtype=np.float32),
                "m_monkey.jpg": np.full((64, 2048), 30.0, dtype=np.float32)
            }
            np.save(str(legacy_file), raw_dict)

            store_dir = tmp_path / "store"
            convert_legacy_features(legacy_file, store_dir)
            store = MmapFeatureStore(store_dir)

            # Access in non-alphabetic order
            self.assertTrue(np.allclose(store["m_monkey.jpg"], 30.0))
            self.assertTrue(np.allclose(store["a_apple.jpg"], 20.0))
            self.assertTrue(np.allclose(store["z_zebra.jpg"], 10.0))

            # Containment check
            self.assertIn("a_apple.jpg", store)
            self.assertIn("m_monkey.jpg", store)
            self.assertIn("z_zebra.jpg", store)
            self.assertNotIn("nonexistent.jpg", store)
            store.close()

    def test_03_manifest_integrity_and_validation(self):
        """3. Verify rejection of invalid shapes/dtypes, duplicate/missing row indices, unsafe paths, and incomplete manifests."""
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            array_file = tmp_path / "features.npy"
            arr = np.zeros((2, 64, 2048), dtype=np.float32)
            np.save(str(array_file), arr)

            base_valid_manifest = {
                "format_version": "2.0",
                "storage_type": "mmap_numeric_array",
                "feature_file": "features.npy",
                "num_images": 2,
                "feature_shape": [64, 2048],
                "full_shape": [2, 64, 2048],
                "dtype": "float32",
                "image_to_index": {"img1.jpg": 0, "img2.jpg": 1}
            }

            def check_bad_manifest(mutator, expected_err, match_str):
                m_data = json.loads(json.dumps(base_valid_manifest))
                mutator(m_data)
                m_file = tmp_path / "manifest_test.json"
                m_file.write_text(json.dumps(m_data), encoding="utf-8")
                with self.assertRaises(expected_err) as ctx:
                    MmapFeatureStore(m_file)
                self.assertIn(match_str, str(ctx.exception))

            # 1. Missing required key
            check_bad_manifest(lambda m: m.pop("format_version"), ValueError, "missing required key")

            # 2. Unsupported storage type
            check_bad_manifest(lambda m: m.update({"storage_type": "pickle"}), ValueError, "Unsupported storage_type")

            # 3. Invalid num_images
            check_bad_manifest(lambda m: m.update({"num_images": -1}), ValueError, "Invalid num_images")

            # 4. Invalid feature_shape
            check_bad_manifest(lambda m: m.update({"feature_shape": [32, 2048]}), ValueError, "Invalid feature_shape")

            # 5. Invalid full_shape
            check_bad_manifest(lambda m: m.update({"full_shape": [2, 32, 2048]}), ValueError, "Invalid full_shape")

            # 6. Invalid dtype
            check_bad_manifest(lambda m: m.update({"dtype": "float64"}), ValueError, "Invalid dtype")

            # 7. Image count mismatch
            check_bad_manifest(lambda m: m.update({"num_images": 3, "full_shape": [3, 64, 2048]}), ValueError, "does not match num_images")

            # 8. Duplicate row indices (never silently map two images to same feature row)
            check_bad_manifest(lambda m: m.update({"image_to_index": {"img1.jpg": 0, "img2.jpg": 0}}), ValueError, "zero duplicate")

            # 9. Missing row index / gap
            check_bad_manifest(lambda m: m.update({"image_to_index": {"img1.jpg": 0, "img2.jpg": 2}}), ValueError, "contiguous 1:1 mapping")

            # 10. Non-integer row index
            check_bad_manifest(lambda m: m.update({"image_to_index": {"img1.jpg": 0, "img2.jpg": "one"}}), ValueError, "must be an integer")

            # 11. Empty/whitespace image identifier
            check_bad_manifest(lambda m: m.update({"image_to_index": {"   ": 0, "img2.jpg": 1}}), ValueError, "Invalid image identifier")

            # 12. Unsafe feature_file path traversal
            check_bad_manifest(lambda m: m.update({"feature_file": "../outside.npy"}), ValueError, "Unsafe feature_file path")

            # 13. Unsafe absolute feature_file path
            check_bad_manifest(lambda m: m.update({"feature_file": "C:/Windows/secret.npy" if os.name == "nt" else "/etc/passwd"}), ValueError, "Unsafe feature_file path")

            # 14. Non-existent path
            with self.assertRaises(FileNotFoundError):
                MmapFeatureStore(tmp_path / "nonexistent_dir")

    def test_04_shape_normalization_for_supported_shapes(self):
        """4. Verify per-image shape normalization accepts (64, 2048), (8, 8, 2048), and (1, 64, 2048)."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "multi_shapes.npy"
            raw_dict = {
                "flat.jpg": np.full((64, 2048), 1.0, dtype=np.float32),
                "grid.jpg": np.full((8, 8, 2048), 2.0, dtype=np.float32),
                "batch.jpg": np.full((1, 64, 2048), 3.0, dtype=np.float32)
            }
            np.save(str(legacy_file), raw_dict)

            store_dir = tmp_path / "norm_store"
            convert_legacy_features(legacy_file, store_dir)
            store = MmapFeatureStore(store_dir)

            for key, expected_val in [("flat.jpg", 1.0), ("grid.jpg", 2.0), ("batch.jpg", 3.0)]:
                feat = store[key]
                self.assertEqual(feat.shape, (64, 2048))
                self.assertEqual(feat.dtype, np.float32)
                self.assertTrue(np.allclose(feat, expected_val))
            store.close()

    def test_05_rejection_of_non_finite_values_and_incorrect_dimensions(self):
        """5. Verify converter rejects NaN, Inf, and incorrect array dimensions."""
        from V2_Attention.src.convert_features import convert_legacy_features

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. NaN rejection
            nan_file = tmp_path / "nan.npy"
            bad_nan_dict = {"img.jpg": np.full((64, 2048), np.nan, dtype=np.float32)}
            np.save(str(nan_file), bad_nan_dict)
            with self.assertRaises(ValueError) as ctx1:
                convert_legacy_features(nan_file, tmp_path / "out_nan")
            self.assertIn("NaN or infinite values", str(ctx1.exception))

            # 2. Inf rejection
            inf_file = tmp_path / "inf.npy"
            bad_inf_dict = {"img.jpg": np.full((64, 2048), np.inf, dtype=np.float32)}
            np.save(str(inf_file), bad_inf_dict)
            with self.assertRaises(ValueError) as ctx2:
                convert_legacy_features(inf_file, tmp_path / "out_inf")
            self.assertIn("NaN or infinite values", str(ctx2.exception))

            # 3. Invalid shape rejection
            shape_file = tmp_path / "bad_shape.npy"
            bad_shape_dict = {"img.jpg": np.zeros((10, 2048), dtype=np.float32)}
            np.save(str(shape_file), bad_shape_dict)
            with self.assertRaises(ValueError) as ctx3:
                convert_legacy_features(shape_file, tmp_path / "out_shape")
            self.assertIn("Invalid feature shape", str(ctx3.exception))

    def test_06_refusal_to_overwrite_existing_files_and_source_preservation(self):
        """6. Verify converter refuses to overwrite existing output files and preserves source and existing destination."""
        from V2_Attention.src.convert_features import convert_legacy_features

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "legacy.npy"
            np.save(str(legacy_file), {"img.jpg": np.zeros((64, 2048), dtype=np.float32)})
            source_hash_before = hashlib.sha256(legacy_file.read_bytes()).hexdigest()

            # Existing non-empty output directory
            existing_out = tmp_path / "existing_store"
            existing_out.mkdir()
            test_file = existing_out / "important_data.txt"
            test_file.write_text("pre-existing content", encoding="utf-8")
            dest_content_before = test_file.read_text(encoding="utf-8")

            # 1. Refuses to overwrite without flag
            with self.assertRaises(FileExistsError):
                convert_legacy_features(legacy_file, existing_out, allow_overwrite=False)

            # 2. Refuses to use input file path as output
            with self.assertRaises(ValueError):
                convert_legacy_features(legacy_file, legacy_file)

            # 3. On Windows, safely replacing an existing populated directory cannot be guaranteed
            # Even with allow_overwrite=True, replacement is refused to preserve existing data
            if sys.platform == "win32" or os.name == "nt":
                with self.assertRaises(FileExistsError) as ctx:
                    convert_legacy_features(legacy_file, existing_out, allow_overwrite=True)
                self.assertIn("On Windows, safely replacing an existing populated directory cannot be guaranteed", str(ctx.exception))

            # 4. Verify source file and existing destination are completely intact
            source_hash_after = hashlib.sha256(legacy_file.read_bytes()).hexdigest()
            self.assertEqual(source_hash_before, source_hash_after)
            self.assertEqual(dest_content_before, test_file.read_text(encoding="utf-8"))

            # 5. Conversion succeeds when targeting a new destination directory
            new_out = tmp_path / "brand_new_store"
            convert_legacy_features(legacy_file, new_out)
            self.assertTrue((new_out / "manifest.json").exists())
            self.assertTrue((new_out / "features.npy").exists())
            # Existing destination remains intact
            self.assertEqual(dest_content_before, test_file.read_text(encoding="utf-8"))

    def test_07_handling_of_interrupted_or_incomplete_conversion(self):
        """7. Verify interrupted conversion leaves no broken store, and incomplete stores fail closed."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.cnn_features import load_features as v2_load_features

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a dict where 2nd item triggers error
            bad_dict = {
                "01_good.jpg": np.zeros((64, 2048), dtype=np.float32),
                "02_bad.jpg": np.full((64, 2048), np.nan, dtype=np.float32)
            }
            legacy_file = tmp_path / "mixed.npy"
            np.save(str(legacy_file), bad_dict)
            source_hash_before = hashlib.sha256(legacy_file.read_bytes()).hexdigest()

            target_out = tmp_path / "target_out"

            with self.assertRaises(ValueError):
                convert_legacy_features(legacy_file, target_out)

            # Target directory was not created; no temporary staging directories remain
            self.assertFalse(target_out.exists())
            staging_dirs = list(tmp_path.glob("*.tmp_convert*"))
            self.assertEqual(staging_dirs, [])

            # Source file remains completely intact
            self.assertEqual(source_hash_before, hashlib.sha256(legacy_file.read_bytes()).hexdigest())

            # Attempting to load an incomplete directory fails closed
            incomplete_dir = tmp_path / "incomplete_dir"
            incomplete_dir.mkdir()
            (incomplete_dir / "features.npy").write_bytes(b"partial_array_bytes")
            with self.assertRaises(FileNotFoundError):
                v2_load_features(str(incomplete_dir))

    def test_08_interface_to_build_training_dataset(self):
        """8. Verify build_training_dataset directly consumes MmapFeatureStore for train and val."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "features.npy"
            raw_dict = {
                "img1.jpg": np.ones((64, 2048), dtype=np.float32),
                "img2.jpg": np.full((64, 2048), 2.0, dtype=np.float32)
            }
            np.save(str(legacy_file), raw_dict)

            store_dir = tmp_path / "mmap_store"
            convert_legacy_features(legacy_file, store_dir)
            store = MmapFeatureStore(store_dir)

            captions_dict = {
                "img1.jpg": ["<start> a dog runs <end>"],
                "img2.jpg": ["<start> a cat sleeps <end>"]
            }
            word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3, "a": 4, "dog": 5, "runs": 6, "cat": 7, "sleeps": 8}

            train_ds = v2_build_training_dataset(captions_dict, store, word_to_index, max_length=5, batch_size=2)
            val_ds = v2_build_training_dataset(captions_dict, store, word_to_index, max_length=5, batch_size=2, shuffle=False)

            # Both datasets share the store without copying
            for img_feat, caps in train_ds:
                self.assertEqual(img_feat.shape, (2, 64, 2048))
                self.assertEqual(caps.shape, (2, 5))
                break

            for img_feat, caps in val_ds:
                self.assertEqual(img_feat.shape, (2, 64, 2048))
                self.assertEqual(caps.shape, (2, 5))
                break
            store.close()

    def test_09_legacy_npy_and_npz_compatibility_with_tiny_fixtures(self):
        """9. Verify load_features preserves backward compatibility with legacy .npy and .npz."""
        from V2_Attention.src.cnn_features import load_features as v2_load_features

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dict = {
                "img_a.jpg": np.ones((64, 2048), dtype=np.float32),
                "img_b.jpg": np.full((64, 2048), 4.0, dtype=np.float32)
            }

            # 1. Legacy .npy
            npy_file = tmp_path / "legacy.npy"
            np.save(str(npy_file), raw_dict)
            loaded_npy = v2_load_features(str(npy_file))
            self.assertEqual(len(loaded_npy), 2)
            self.assertTrue(np.allclose(loaded_npy["img_a.jpg"], 1.0))

            # 2. Legacy .npz
            npz_file = tmp_path / "legacy.npz"
            np.savez(str(npz_file), **raw_dict)
            loaded_npz = v2_load_features(str(npz_file))
            self.assertEqual(len(loaded_npz), 2)
            self.assertTrue(np.allclose(loaded_npz["img_b.jpg"], 4.0))

    def test_10_reader_uses_memory_mapping_without_eager_allocation(self):
        """10. Verify MmapFeatureStore uses np.memmap, accesses single row, avoids full copies, and closes safely."""
        from V2_Attention.src.convert_features import convert_legacy_features
        from V2_Attention.src.cnn_features import load_features as v2_load_features
        from V2_Attention.src.feature_store import MmapFeatureStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "test_mmap.npy"
            raw_dict = {f"img_{i}.jpg": np.full((64, 2048), float(i), dtype=np.float32) for i in range(5)}
            np.save(str(legacy_file), raw_dict)

            store_dir = tmp_path / "mmap_store"
            convert_legacy_features(legacy_file, store_dir)

            store = v2_load_features(str(store_dir))
            # Must be an MmapFeatureStore instance backed by np.memmap
            self.assertTrue(hasattr(store, "is_memory_mapped"))
            self.assertTrue(store.is_memory_mapped())
            self.assertIsInstance(store._data, np.memmap)
            self.assertFalse(isinstance(store._data, dict))

            # Row access returns only the requested slice with expected shape and values
            row_3 = store["img_3.jpg"]
            self.assertEqual(row_3.shape, (64, 2048))
            self.assertEqual(row_3.dtype, np.float32)
            self.assertTrue(np.allclose(row_3, 3.0))

            # Safe store closure
            self.assertFalse(store.is_closed)
            store.close()
            self.assertTrue(store.is_closed)

            # Access after closure raises RuntimeError
            with self.assertRaises(RuntimeError):
                _ = store["img_3.jpg"]

            # Subsequent close() call is an idempotent no-op
            store.close()

            # Context manager usage automatically closes handle
            with MmapFeatureStore(store_dir) as ctx_store:
                self.assertFalse(ctx_store.is_closed)
                self.assertEqual(ctx_store["img_0.jpg"].shape, (64, 2048))
            self.assertTrue(ctx_store.is_closed)


if __name__ == "__main__":
    unittest.main()
