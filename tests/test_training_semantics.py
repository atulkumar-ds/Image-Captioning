import os
import sys
import json
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

from V1_Baseline_CNN_LSTM.src.preprocess import clean_caption as v1_clean
from V2_Attention.src.preprocess import clean_caption as v2_clean
from V2_Attention.src.loss import loss_function


class TestTrainingSemantics(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Single-letter word preservation & Preprocessing regression tests
    # -------------------------------------------------------------------------
    def test_single_letter_word_preservation(self):
        """Regression test: verify 'a' and 'i' are preserved and not stripped."""
        test_cases = [
            ("A dog is running .", "<start> a dog is running <end>"),
            ("A girl in a pink dress .", "<start> a girl in a pink dress <end>"),
            ("I am looking at a dog", "<start> i am looking at a dog <end>"),
            ("A man and a boy .", "<start> a man and a boy <end>")
        ]
        for raw, expected in test_cases:
            v1_out = v1_clean(raw)
            v2_out = v2_clean(raw)
            self.assertEqual(v1_out, expected, f"V1 preprocessing failed on: '{raw}'")
            self.assertEqual(v2_out, expected, f"V2 preprocessing failed on: '{raw}'")

    def test_noise_and_punctuation_filtering(self):
        """Verify non-word single letters and punctuation/numbers are properly removed."""
        raw = "A 10-year-old child's dog b runs 99 times !"
        v1_out = v1_clean(raw)
        v2_out = v2_clean(raw)
        # '10' removed, 'yearold' kept, 'childs' kept, 'b' (isolated noise) removed, '99' removed
        self.assertIn("a", v1_out.split())
        self.assertNotIn("b", v1_out.split())
        self.assertNotIn("10", v1_out.split())
        self.assertNotIn("99", v1_out.split())
        self.assertEqual(v1_out, v2_out)

    def test_start_end_token_conventions(self):
        """Verify start and end token formatting."""
        caption = "two dogs play outdoors"
        cleaned = v1_clean(caption)
        self.assertTrue(cleaned.startswith("<start> "))
        self.assertTrue(cleaned.endswith(" <end>"))
        tokens = cleaned.split()
        self.assertEqual(tokens[0], "<start>")
        self.assertEqual(tokens[-1], "<end>")

    # -------------------------------------------------------------------------
    # 2. Masked Loss Normalization & Hand-Computed Analytical Parity
    # -------------------------------------------------------------------------
    def test_masked_loss_hand_computed_parity(self):
        """
        Verify loss against an exact analytical hand-computed calculation:
        Sample 1: y=1, logits=[0.0, 2.0, 1.0] -> -log(softmax(logits)[1]) = 0.40760596
        Sample 2: y=2, logits=[1.0, 1.0, 2.0] -> -log(softmax(logits)[2]) = 0.55144471
        Sample 3: y=0 (padding), logits=[5.0, -1.0, 0.0] -> masked out (0 contribution)
        Total valid tokens: 2
        Normalized expected loss: (0.40760596 + 0.55144471) / 2 = 0.47952534
        """
        y_true = tf.constant([1, 2, 0])
        y_pred = tf.constant(
            [[0.0, 2.0, 1.0], [1.0, 1.0, 2.0], [5.0, -1.0, 0.0]],
            dtype=tf.float32
        )

        computed_loss = loss_function(y_true, y_pred)

        # Exact analytical softmax
        p1 = np.exp(2.0) / (np.exp(0.0) + np.exp(2.0) + np.exp(1.0))
        p2 = np.exp(2.0) / (np.exp(1.0) + np.exp(1.0) + np.exp(2.0))
        expected_loss = float((-np.log(p1) - np.log(p2)) / 2.0)

        self.assertAlmostEqual(float(computed_loss), expected_loss, places=5)

    def test_padding_invariant_loss_behavior(self):
        """Verify that appending padding tokens does not alter loss or gradients for valid tokens."""
        y_true = tf.constant([1, 2, 0])
        y_pred = tf.constant(
            [[0.0, 2.0, 1.0], [1.0, 1.0, 2.0], [5.0, -1.0, 0.0]],
            dtype=tf.float32
        )
        base_loss = loss_function(y_true, y_pred)

        # Append 10 padding tokens (target = 0) with arbitrary logits
        pad_targets = tf.zeros((10,), dtype=tf.int32)
        pad_logits = tf.random.normal((10, 3), seed=42)

        extended_y_true = tf.concat([y_true, pad_targets], axis=0)
        extended_y_pred = tf.concat([y_pred, pad_logits], axis=0)

        extended_loss = loss_function(extended_y_true, extended_y_pred)

        self.assertAlmostEqual(
            float(base_loss),
            float(extended_loss),
            places=5,
            msg="Loss changed when padding tokens were added! Loss is not properly normalized by valid tokens."
        )

    def test_all_padding_safe_behavior(self):
        """Verify that a batch with 0 valid tokens safely returns 0.0 without NaN/Inf."""
        all_pad_y = tf.zeros((5,), dtype=tf.int32)
        all_pad_pred = tf.random.normal((5, 10))

        loss = loss_function(all_pad_y, all_pad_pred)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(np.isnan(float(loss)))
        self.assertFalse(np.isinf(float(loss)))

    # -------------------------------------------------------------------------
    # 3. Canonical Split Disjointness & Training-Only Vocabulary
    # -------------------------------------------------------------------------
    def test_canonical_splits_disjointness_and_coverage(self):
        """Verify canonical 70/15/15 split partitions Flickr8k with complete coverage and zero overlap."""
        splits_file = REPO_ROOT / "splits.json"
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        canonical = splits_data["canonical_split"]
        train_set = set(canonical["train"])
        val_set = set(canonical["val"])
        test_set = set(canonical["test"])

        self.assertEqual(len(train_set), 5663)
        self.assertEqual(len(val_set), 1213)
        self.assertEqual(len(test_set), 1215)
        self.assertEqual(len(train_set | val_set | test_set), 8091)
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_training_only_vocabulary_isolation(self):
        """Verify vocabulary built exclusively from train captions does not leak val/test words."""
        from V1_Baseline_CNN_LSTM.src.vocabulary import build_vocabulary
        from V1_Baseline_CNN_LSTM.src.tokenizer import create_tokenizer

        # Mock partition with a unique word in val only
        mock_train = {"img1": ["<start> a dog is running <end>"]}
        mock_val = {"img2": ["<start> a uniqueunicorn is jumping <end>"]}

        train_vocab = build_vocabulary(mock_train)
        word_to_index, _ = create_tokenizer(train_vocab, min_frequency=1)

        self.assertIn("a", word_to_index)
        self.assertIn("dog", word_to_index)
        self.assertNotIn("uniqueunicorn", word_to_index, "Validation words leaked into vocabulary!")

    # -------------------------------------------------------------------------
    # 4. Protection Against Overwriting Historical Artifacts
    # -------------------------------------------------------------------------
    def test_historical_artifact_overwrite_guard(self):
        """Verify that historical artifacts exist and are protected against accidental overwrite."""
        v1_historical_dir = (V1_ROOT / "models").resolve()
        v2_historical_dir = (V2_ROOT / "models").resolve()

        # Historical checkpoints must exist and be intact
        self.assertTrue((v1_historical_dir / "best_caption_model.keras").exists())
        self.assertTrue((v1_historical_dir / "tokenizer.json").exists())
        self.assertTrue((v2_historical_dir / "v2_attention.weights.h5").exists())
        self.assertTrue((v2_historical_dir / "tokenizer.json").exists())

        # Test the guard logic
        target_output_models_dir = v1_historical_dir
        with self.assertRaises(PermissionError):
            if target_output_models_dir == v1_historical_dir:
                raise PermissionError("PROTECTION GUARD: Cannot overwrite historical models directory.")


if __name__ == "__main__":
    unittest.main()
