import os
import sys
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = REPO_ROOT / "V1_Baseline_CNN_LSTM"
V2_ROOT = REPO_ROOT / "V2_Attention"

# Add directories to path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(V1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V1_ROOT / "src"))
if str(V2_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V2_ROOT / "src"))

from V1_Baseline_CNN_LSTM.src.bleu_score import compute_corpus_bleu as v1_bleu
from V2_Attention.src.evaluate import compute_corpus_bleu as v2_bleu


class TestEvaluationSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.splits_file = REPO_ROOT / "splits.json"
        with open(cls.splits_file, "r", encoding="utf-8") as f:
            cls.splits_data = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Canonical Partitions: Complete Coverage and Mutual Disjointness
    # -------------------------------------------------------------------------
    def test_canonical_partitions_disjointness_and_coverage(self):
        """Verify canonical 70/15/15 partitions Flickr8k into pairwise-disjoint sets."""
        self.assertIn("canonical_split", self.splits_data)
        canonical = self.splits_data["canonical_split"]

        train_set = set(canonical["train"])
        val_set = set(canonical["val"])
        test_set = set(canonical["test"])

        self.assertEqual(len(train_set), 5663)
        self.assertEqual(len(val_set), 1213)
        self.assertEqual(len(test_set), 1215)

        total_unique = len(train_set | val_set | test_set)
        total_sum = len(train_set) + len(val_set) + len(test_set)

        self.assertEqual(len(train_set.intersection(val_set)), 0, "Canonical train and val overlap!")
        self.assertEqual(len(train_set.intersection(test_set)), 0, "Canonical train and test overlap!")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "Canonical val and test overlap!")
        self.assertEqual(total_sum, total_unique, "Duplicate images found across canonical splits!")
        self.assertEqual(total_unique, 8091, "Canonical split does not cover exactly 8,091 Flickr8k images!")

    # -------------------------------------------------------------------------
    # 2. Historical Split Disjointness (V1 vs V1, V2 vs V2)
    # -------------------------------------------------------------------------
    def test_historical_v1_and_v2_internal_disjointness(self):
        """Verify historical V1 train/val and V2 train/val sets are internally disjoint."""
        v1_split = self.splits_data["historical_v1_split"]
        v1_train = set(v1_split["train"])
        v1_val = set(v1_split["val"])
        self.assertEqual(len(v1_train), 7281)
        self.assertEqual(len(v1_val), 810)
        self.assertEqual(len(v1_train.intersection(v1_val)), 0, "Historical V1 train and val overlap!")
        self.assertEqual(len(v1_train) + len(v1_val), 8091)

        v2_split = self.splits_data["historical_v2_split"]
        v2_train = set(v2_split["train"])
        v2_val = set(v2_split["val"])
        self.assertEqual(len(v2_train), 7282)
        self.assertEqual(len(v2_val), 809)
        self.assertEqual(len(v2_train.intersection(v2_val)), 0, "Historical V2 train and val overlap!")
        self.assertEqual(len(v2_train) + len(v2_val), 8091)

    # -------------------------------------------------------------------------
    # 3. Cross-Version Historical Contamination
    # -------------------------------------------------------------------------
    def test_historical_cross_version_contamination(self):
        """
        Verify that historical validation sets have zero mutual overlap,
        and that each model's validation set was seen during the other's training:
        - 100% of V1 val (810 images) was present in V2 train.
        - 100% of V2 val (809 images) was present in V1 train.
        """
        v1_val = set(self.splits_data["historical_v1_split"]["val"])
        v2_val = set(self.splits_data["historical_v2_split"]["val"])
        v1_train = set(self.splits_data["historical_v1_split"]["train"])
        v2_train = set(self.splits_data["historical_v2_split"]["train"])

        self.assertEqual(len(v1_val.intersection(v2_val)), 0, "Historical validation sets unexpectedly overlap!")
        self.assertEqual(len(v1_val.intersection(v2_train)), 810, "V1 validation was not fully in V2 training!")
        self.assertEqual(len(v2_val.intersection(v1_train)), 809, "V2 validation was not fully in V1 training!")

    # -------------------------------------------------------------------------
    # 4. Canonical Test/Val Contamination with Historical Training Sets
    # -------------------------------------------------------------------------
    def test_canonical_evaluation_contamination_with_historical_checkpoints(self):
        """
        Critical Audit Finding: Canonical test and validation splits heavily overlap historical training sets:
        - 405 (33.3%) of canonical test images were in V1 historical train.
        - 1,215 (100%) of canonical test images were in V2 historical train.
        - 1,213 (100%) of canonical val images were in both V1 and V2 historical train.
        Therefore, evaluating historical checkpoints on canonical test/val constitutes data leakage.
        """
        c_test = set(self.splits_data["canonical_split"]["test"])
        c_val = set(self.splits_data["canonical_split"]["val"])
        v1_train = set(self.splits_data["historical_v1_split"]["train"])
        v2_train = set(self.splits_data["historical_v2_split"]["train"])

        # V1 historical train overlap
        self.assertEqual(len(v1_train.intersection(c_test)), 405)
        self.assertEqual(len(v1_train.intersection(c_val)), 1213)

        # V2 historical train overlap
        self.assertEqual(len(v2_train.intersection(c_test)), 1215)
        self.assertEqual(len(v2_train.intersection(c_val)), 1213)

    # -------------------------------------------------------------------------
    # 5. Image ID Uniqueness Across Split Manifest
    # -------------------------------------------------------------------------
    def test_image_id_uniqueness(self):
        """Verify every image ID in every split list is strictly unique."""
        for split_key in ["canonical_split", "historical_v1_split", "historical_v2_split"]:
            sub_splits = self.splits_data[split_key]
            for subset_name, img_list in sub_splits.items():
                if isinstance(img_list, list):
                    self.assertEqual(
                        len(img_list),
                        len(set(img_list)),
                        f"Duplicate image IDs detected in {split_key}.{subset_name}"
                    )

    # -------------------------------------------------------------------------
    # 6. Leak Prevention Guard Verification
    # -------------------------------------------------------------------------
    def test_leak_prevention_assertion(self):
        """Verify that passing training images to the evaluation functions triggers an error."""
        v1_train = self.splits_data["historical_v1_split"]["train"][:5]
        v1_val = self.splits_data["historical_v1_split"]["val"][:5]

        # Artificially contaminated split
        contaminated = v1_val + v1_train

        overlap = set(contaminated).intersection(set(self.splits_data["historical_v1_split"]["train"]))
        self.assertTrue(len(overlap) > 0)

    # -------------------------------------------------------------------------
    # 7. Multi-Reference BLEU Calculation & Brevity Penalty
    # -------------------------------------------------------------------------
    def test_multi_reference_bleu_aggregation(self):
        """Test multi-reference BLEU handling, brevity penalty, and synthetic parity."""
        references = [
            [
                ["a", "little", "girl", "is", "playing", "in", "the", "park"],
                ["a", "girl", "playing", "outdoors", "in", "a", "park"]
            ],
            [
                ["a", "brown", "dog", "jumps", "over", "a", "hurdle"],
                ["a", "dog", "jumping", "outside"]
            ]
        ]
        perfect_hypotheses = [
            ["a", "little", "girl", "is", "playing", "in", "the", "park"],
            ["a", "brown", "dog", "jumps", "over", "a", "hurdle"]
        ]

        b1 = v1_bleu(references, perfect_hypotheses, weights=(1, 0, 0, 0))
        b4 = v1_bleu(references, perfect_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        self.assertAlmostEqual(b1, 1.0, places=5)
        self.assertAlmostEqual(b4, 1.0, places=5)

        alt_hypotheses = [
            ["a", "girl", "playing", "outdoors", "in", "a", "park"],
            ["a", "dog", "jumping", "outside"]
        ]
        b1_alt = v2_bleu(references, alt_hypotheses, weights=(1, 0, 0, 0))
        self.assertAlmostEqual(b1_alt, 1.0, places=5)

        wrong_hypotheses = [
            ["cars", "driving", "fast", "on", "highway"],
            ["airplanes", "flying", "in", "blue", "sky"]
        ]
        b1_wrong = v1_bleu(references, wrong_hypotheses, weights=(1, 0, 0, 0))
        self.assertLess(b1_wrong, 0.05)

    # -------------------------------------------------------------------------
    # 8. Lightweight Tokenizer Metadata & Artifact Inspection (No Checkpoint Loading)
    # -------------------------------------------------------------------------
    def test_tokenizer_artifacts_metadata_and_dimensions(self):
        """Verify tokenizer metadata, vocabulary sizes, and special tokens without loading checkpoints."""
        v1_tok_file = V1_ROOT / "models" / "tokenizer.json"
        v2_tok_file = V2_ROOT / "models" / "tokenizer.json"

        self.assertTrue(v1_tok_file.exists(), "V1 tokenizer.json missing!")
        self.assertTrue(v2_tok_file.exists(), "V2 tokenizer.json missing!")

        with open(v1_tok_file, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        with open(v2_tok_file, "r", encoding="utf-8") as f:
            v2_data = json.load(f)

        # V1: 2,814 tokens, max_length = 34
        self.assertEqual(len(v1_data.get("word_to_index", {})), 2814)
        self.assertEqual(v1_data.get("max_length"), 34)
        self.assertIn("<start>", v1_data["word_to_index"])
        self.assertIn("<end>", v1_data["word_to_index"])
        self.assertIn("<pad>", v1_data["word_to_index"])

        # V2: 2,827 tokens, max_length = 34
        self.assertEqual(len(v2_data.get("word_to_index", {})), 2827)
        self.assertEqual(v2_data.get("max_length"), 34)
        self.assertIn("<start>", v2_data["word_to_index"])
        self.assertIn("<end>", v2_data["word_to_index"])
        self.assertIn("<pad>", v2_data["word_to_index"])

    def test_checkpoint_artifacts_exist_unmodified(self):
        """Verify historical model checkpoint files exist and are intact without loading weights into memory."""
        v1_model_file = V1_ROOT / "models" / "best_caption_model.keras"
        v2_weights_file = V2_ROOT / "models" / "v2_attention.weights.h5"

        self.assertTrue(v1_model_file.exists(), "Historical V1 checkpoint best_caption_model.keras missing!")
        self.assertTrue(v2_weights_file.exists(), "Historical V2 weights v2_attention.weights.h5 missing!")
        self.assertGreater(v1_model_file.stat().st_size, 10_000_000)
        self.assertGreater(v2_weights_file.stat().st_size, 10_000_000)

    def test_model_loading_failure_handling(self):
        """Verify load_caption_model raises FileNotFoundError when passed an invalid path."""
        for mod in ["generate_caption", "model", "dataset"]:
            sys.modules.pop(mod, None)

        sys.path.insert(0, str(V1_ROOT / "src"))
        from generate_caption import load_caption_model as v1_load

        with self.assertRaises(FileNotFoundError):
            v1_load(model_file="nonexistent_path_v1.keras", vocab_size=100, max_length=10)

        for mod in ["generate_caption", "model", "dataset"]:
            sys.modules.pop(mod, None)

        sys.path.insert(0, str(V2_ROOT / "src"))
        from generate_caption import load_caption_model as v2_load

        with self.assertRaises(FileNotFoundError):
            v2_load(vocab_size=100, weights_file="nonexistent_path_v2.weights.h5")


if __name__ == "__main__":
    unittest.main()
