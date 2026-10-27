import os
import sys
import json
import math
import argparse
import hashlib
from pathlib import Path
from collections import Counter

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

SRC_DIR = Path(__file__).resolve().parent
V2_ROOT = SRC_DIR.parent
REPO_ROOT = V2_ROOT.parent

# Set working directory to V2_ROOT so legacy dataset.py relative paths resolve
os.chdir(V2_ROOT)

# Ensure local imports work regardless of execution location
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from V2_Attention.src.dataset import load_captions, clean_all_captions
    from V2_Attention.src.generate_caption import (
        load_tokenizer,
        load_cnn,
        load_caption_model,
        generate_caption
    )
except ImportError:
    from dataset import load_captions, clean_all_captions
    from generate_caption import (
        load_tokenizer,
        load_cnn,
        load_caption_model,
        generate_caption
    )

HISTORICAL_WEIGHTS_FILE = (V2_ROOT / "models" / "v2_attention.weights.h5").resolve()
HISTORICAL_TOKENIZER_FILE = (V2_ROOT / "models" / "tokenizer.json").resolve()
HISTORICAL_MODELS_DIR = (V2_ROOT / "models").resolve()
DEFAULT_IMAGE_DIR = REPO_ROOT / "V1_Baseline_CNN_LSTM" / "data" / "images"
SPLITS_FILE = REPO_ROOT / "splits.json"

HISTORICAL_V2_CHECKPOINT_HASHES = {
    "7ecbb57319a9bfcd2b29ba15708e4814547bf6f5a9b8b3137d66386a872eb838": "v2_attention.weights.h5",
}
HISTORICAL_V2_TOKENIZER_HASH = "fcb52a1cafd310f2920375bb887bbcba26e7a1b0bd6a6844458078de112abb6d"


def compute_file_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _compute_ngrams(sequence, n):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


def compute_corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_method="method1"):
    """
    Pure-Python fallback for corpus BLEU matching NLTK corpus_bleu
    with Papineni et al. (2002) brevity penalty and Chen & Cherry (2014) method 1 smoothing.
    """
    max_order = len(weights)
    numerator = [0] * max_order
    denominator = [0] * max_order
    hyp_lengths = 0
    ref_lengths = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len

        ref_lens = [len(ref) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
        ref_lengths += closest_ref_len

        for i in range(max_order):
            n = i + 1
            hyp_ngrams = Counter(_compute_ngrams(hypothesis, n))
            max_ref_counts = Counter()
            for ref in references:
                ref_ngrams = Counter(_compute_ngrams(ref, n))
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            for ngram, hyp_count in hyp_ngrams.items():
                numerator[i] += min(hyp_count, max_ref_counts[ngram])
            denominator[i] += max(len(hypothesis) - n + 1, 0)

    if hyp_lengths == 0:
        return 0.0

    if hyp_lengths > ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_lengths) / hyp_lengths)

    p_log_sum = 0.0
    for i, weight in enumerate(weights):
        if weight == 0.0:
            continue
        num = numerator[i]
        denom = denominator[i]
        if denom == 0:
            return 0.0

        if num > 0:
            p = float(num) / float(denom)
        else:
            if smoothing_method == "method1":
                p = 0.1 / float(denom)
            else:
                p = 0.0
        if p <= 0.0:
            return 0.0
        p_log_sum += weight * math.log(p)

    return float(bp * math.exp(p_log_sum))


def compute_split_manifest_fingerprint(split_data, schema_version="canonical_v1"):
    """
    Computes a deterministic canonical SHA-256 fingerprint for the complete partition structure.
    Covers:
      - schema_version
      - sorted training image IDs
      - sorted validation image IDs
      - sorted test image IDs
    Serialized canonically to UTF-8 JSON with sorted keys and compact separators.
    """
    canonical_rep = {
        "schema_version": schema_version,
        "test": sorted(split_data.get("test", [])),
        "train": sorted(split_data.get("train", [])),
        "val": sorted(split_data.get("val", []))
    }
    canonical_bytes = json.dumps(canonical_rep, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def validate_split_manifest_structure(manifest_data, split_key="canonical_split"):
    """
    Rigorously validates the split manifest structure:
    1. Verifies manifest_data is a dictionary.
    2. Verifies split_key exists and is a dictionary.
    3. Verifies required partition fields ('train', 'val', and 'test' for canonical; 'train' and 'val' for historical).
    4. Verifies partitions are lists of strings.
    5. Detects duplicate image IDs within any partition.
    6. Detects cross-partition overlaps between train, val, and test.
    """
    if not isinstance(manifest_data, dict):
        raise ValueError("MALFORMED SPLIT MANIFEST: Root manifest must be a JSON object/dictionary.")

    if split_key not in manifest_data:
        raise ValueError(f"MALFORMED SPLIT MANIFEST: Required split key '{split_key}' not found in manifest.")

    split_data = manifest_data[split_key]
    if not isinstance(split_data, dict):
        raise ValueError(f"MALFORMED SPLIT MANIFEST: Split '{split_key}' must be a dictionary.")

    # Check required fields
    required_fields = ["train", "val"]
    if "canonical" in split_key:
        required_fields.append("test")

    for field in required_fields:
        if field not in split_data:
            raise ValueError(f"MALFORMED SPLIT MANIFEST: Split '{split_key}' missing required partition '{field}'.")
        if not isinstance(split_data[field], list):
            raise ValueError(f"MALFORMED SPLIT MANIFEST: Partition '{field}' in '{split_key}' must be a list.")

    # Check for duplicate image IDs within each partition
    for field in required_fields:
        items = split_data[field]
        if len(items) != len(set(items)):
            raise ValueError(f"DUPLICATE IMAGE IDS DETECTED: Partition '{field}' in '{split_key}' contains duplicate image IDs.")

    # Check for cross-partition overlaps
    train_set = set(split_data.get("train", []))
    val_set = set(split_data.get("val", []))
    test_set = set(split_data.get("test", []))

    overlap_train_val = train_set.intersection(val_set)
    if overlap_train_val:
        raise ValueError(f"CROSS-PARTITION OVERLAP DETECTED: {len(overlap_train_val)} images overlap between train and val in '{split_key}'.")

    if "canonical" in split_key:
        overlap_train_test = train_set.intersection(test_set)
        if overlap_train_test:
            raise ValueError(f"CROSS-PARTITION OVERLAP DETECTED: {len(overlap_train_test)} images overlap between train and test in '{split_key}'.")

        overlap_val_test = val_set.intersection(test_set)
        if overlap_val_test:
            raise ValueError(f"CROSS-PARTITION OVERLAP DETECTED: {len(overlap_val_test)} images overlap between val and test in '{split_key}'.")


def save_experiment_config_safely(config_data, target_path):
    """
    Atomically writes experiment configuration JSON to prevent partial or interrupted
    metadata updates from corrupting the experiment state.
    """
    target_path = Path(target_path).resolve()
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def resolve_v2_evaluation_context(
    weights_file=None,
    tokenizer_file=None,
    trained_on=None,
    config_file=None,
    split_name=None
):
    """
    Resolves and rigorously validates the model weights, tokenizer, and training split context.

    Rules enforced:
    1. Historical weights (by path or by SHA-256 fingerprint) are IMMUTABLY bound
       to historical_v2_split and models/tokenizer.json.
       Declaring --trained_on canonical, copying historical weights to experiment dirs,
       or using non-historical tokenizers triggers an immediate ValueError.
    2. Non-historical experiment weights MUST be accompanied by an experiment config.json.
       Validates: split_type, model_version, train_split_fingerprint, split_manifest_fingerprint,
       tokenizer_fingerprint, and required checkpoint_fingerprint.
    3. Checkpoint fingerprint MUST be present and verified against file bytes for completed models.
    4. Arbitrary/untracked checkpoints without metadata are rejected to prevent unverified evaluations.
    """
    if weights_file is None:
        wt_path = HISTORICAL_WEIGHTS_FILE
    else:
        wt_path = Path(weights_file).resolve()

    if not wt_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {wt_path}")

    ckpt_hash = compute_file_sha256(wt_path)
    is_historical_hash = ckpt_hash in HISTORICAL_V2_CHECKPOINT_HASHES
    is_historical_path = (
        wt_path == HISTORICAL_WEIGHTS_FILE
        or (wt_path.parent == HISTORICAL_MODELS_DIR and wt_path.name == "v2_attention.weights.h5")
    )
    is_historical = is_historical_hash or is_historical_path

    if is_historical:
        # Enforce historical immutability: CLI cannot declare historical models as canonical,
        # even if the weights file has been copied to another folder.
        if (trained_on is not None and trained_on.lower() not in ("historical_v2", "historical")) or (is_historical_hash and not is_historical_path and trained_on == "canonical"):
            raise ValueError(
                f"SECURITY / LEAK GUARD CONFLICT: Checkpoint '{wt_path.name}' matches the SHA-256 fingerprint of an "
                f"immutable historical artifact trained on 'historical_v2_split'. It cannot be declared as trained on '{trained_on}'. "
                f"Copied historical checkpoints cannot bypass historical evaluation protection."
            )

        if split_name is not None and ("canonical" in split_name.lower()):
            raise ValueError(
                f"SECURITY / LEAK GUARD CONFLICT: Historical checkpoint '{wt_path.name}' was trained on 'historical_v2_split' "
                f"and cannot be evaluated on '{split_name}'. Historical checkpoints must only be evaluated on 'historical_v2_val'."
            )

        training_split_key = "historical_v2_split"
        experiment_split_type = "historical_v2"

        # Validate tokenizer for historical checkpoint
        if tokenizer_file is not None:
            tok_path = Path(tokenizer_file).resolve()
            if not tok_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
            tok_hash = compute_file_sha256(tok_path)
            if tok_hash != HISTORICAL_V2_TOKENIZER_HASH:
                raise ValueError(
                    f"INCOMPATIBLE ARTIFACTS: Historical checkpoint '{wt_path.name}' must be evaluated using its "
                    f"matching historical tokenizer at '{HISTORICAL_TOKENIZER_FILE}'. Received '{tok_path}'."
                )
        else:
            tok_path = HISTORICAL_TOKENIZER_FILE

        config_metadata = {
            "model_version": "V2_Attention",
            "split_type": "historical_v2",
            "vocab_size": 2827,
            "max_length": 34
        }
    else:
        # Non-historical experiment checkpoint: MUST validate config.json metadata
        if config_file is not None:
            cfg_path = Path(config_file).resolve()
        else:
            candidate_1 = wt_path.parent / "config.json"
            candidate_2 = wt_path.parent.parent / "config.json"
            if candidate_1.exists():
                cfg_path = candidate_1
            elif candidate_2.exists():
                cfg_path = candidate_2
            else:
                cfg_path = None

        if cfg_path is None or not cfg_path.exists():
            raise ValueError(
                f"EXPERIMENT METADATA MISSING: Checkpoint '{wt_path}' is not an authorized historical artifact "
                f"and does not have an associated 'config.json' metadata file. Future experiment checkpoints must "
                f"be accompanied by experiment config metadata specifying the training split and architecture."
            )

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config_metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"CORRUPT EXPERIMENT METADATA: Metadata file '{cfg_path}' is malformed or corrupted: {e}")

        if not isinstance(config_metadata, dict):
            raise ValueError(f"INVALID EXPERIMENT METADATA: Metadata in '{cfg_path}' must be a JSON object.")

        cfg_model_ver = config_metadata.get("model_version")
        if cfg_model_ver and "v2" not in cfg_model_ver.lower():
            raise ValueError(
                f"MODEL VERSION MISMATCH: Experiment config '{cfg_path}' specifies model_version '{cfg_model_ver}', "
                f"which is incompatible with V2 Spatial Attention evaluation."
            )

        cfg_split = config_metadata.get("split_type")
        if not cfg_split:
            raise ValueError(f"INVALID EXPERIMENT METADATA: 'split_type' missing in '{cfg_path}'.")

        if trained_on is not None and trained_on.lower() != cfg_split.lower() and f"{trained_on}_split".lower() != cfg_split.lower():
            raise ValueError(
                f"METADATA CONFLICT: CLI argument --trained_on '{trained_on}' contradicts experiment config "
                f"split_type '{cfg_split}' in '{cfg_path}'!"
            )

        experiment_split_type = cfg_split.lower()
        if experiment_split_type in ("canonical", "canonical_split"):
            training_split_key = "canonical_split"
            if split_name is not None and ("historical" in split_name.lower()):
                raise ValueError(
                    f"METADATA CONFLICT: Canonical experiment checkpoint cannot be evaluated on historical split '{split_name}'. "
                    f"Use 'canonical_val' or 'canonical_test'."
                )
        elif experiment_split_type in ("historical", "historical_v2", "historical_v2_split"):
            training_split_key = "historical_v2_split"
        else:
            training_split_key = f"{experiment_split_type}_split" if not experiment_split_type.endswith("_split") else experiment_split_type

        # Resolve tokenizer (portable relative path or CLI override)
        if tokenizer_file is not None:
            tok_path = Path(tokenizer_file).resolve()
        elif "tokenizer_file" in config_metadata:
            raw_tok = Path(config_metadata["tokenizer_file"])
            tok_path = raw_tok.resolve() if raw_tok.is_absolute() else (cfg_path.parent / raw_tok).resolve()
        elif (wt_path.parent / "tokenizer.json").exists():
            tok_path = (wt_path.parent / "tokenizer.json").resolve()
        elif (wt_path.parent.parent / "models" / "tokenizer.json").exists():
            tok_path = (wt_path.parent.parent / "models" / "tokenizer.json").resolve()
        else:
            raise FileNotFoundError(
                f"Unable to locate tokenizer for experiment checkpoint '{wt_path}'. Specify --tokenizer path."
            )

        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")

        # Exact tokenizer identity verification (cannot be bypassed by matching vocab size)
        tok_hash = compute_file_sha256(tok_path)
        expected_tok_fp = config_metadata.get("tokenizer_fingerprint")
        if expected_tok_fp and tok_hash != expected_tok_fp:
            raise ValueError(
                f"TOKENIZER IDENTITY MISMATCH: The supplied tokenizer '{tok_path}' (SHA-256: {tok_hash}) "
                f"does not match the experiment's recorded tokenizer_fingerprint ({expected_tok_fp}). "
                f"A different tokenizer cannot be used even if vocabulary sizes match!"
            )

        # Checkpoint identity verification: MUST be present, valid, and match bytes for completed model evaluation
        expected_ckpt_fp = config_metadata.get("checkpoint_fingerprint")
        if expected_ckpt_fp is None:
            raise ValueError(
                f"CHECKPOINT FINGERPRINT MISSING: Checkpoint '{wt_path.name}' has null or missing 'checkpoint_fingerprint' "
                f"in experiment metadata '{cfg_path}'. Completed model evaluation requires a finalized, verified checkpoint fingerprint."
            )

        if not isinstance(expected_ckpt_fp, str) or len(expected_ckpt_fp) != 64 or not all(c in "0123456789abcdefABCDEF" for c in expected_ckpt_fp):
            raise ValueError(
                f"MALFORMED CHECKPOINT FINGERPRINT: 'checkpoint_fingerprint' in '{cfg_path}' is malformed ({expected_ckpt_fp})."
            )

        if ckpt_hash.lower() != expected_ckpt_fp.lower():
            raise ValueError(
                f"CHECKPOINT IDENTITY MISMATCH: Checkpoint file '{wt_path}' (SHA-256: {ckpt_hash}) "
                f"does not match the recorded checkpoint_fingerprint ({expected_ckpt_fp}) in experiment metadata!"
            )

        # Split manifest and partition identity verification
        manifest_rel = config_metadata.get("split_manifest", config_metadata.get("splits_file"))
        if manifest_rel:
            manifest_path = Path(manifest_rel) if Path(manifest_rel).is_absolute() else (cfg_path.parent / manifest_rel).resolve()
            if not manifest_path.exists():
                manifest_path = SPLITS_FILE.resolve()
        else:
            manifest_path = SPLITS_FILE.resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest file not found: {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            raise ValueError(f"CORRUPT SPLIT MANIFEST: Manifest file '{manifest_path}' is malformed or corrupted: {e}")

        # Validate structural integrity, duplicates, and partition disjointness
        validate_split_manifest_structure(manifest_data, training_split_key)

        split_data = manifest_data[training_split_key]

        # 1. Training partition verification
        actual_train_imgs = split_data.get("train", [])
        actual_train_fp = hashlib.sha256(json.dumps(sorted(actual_train_imgs)).encode("utf-8")).hexdigest()
        expected_train_fp = config_metadata.get("train_split_fingerprint")
        if expected_train_fp and actual_train_fp != expected_train_fp:
            raise ValueError(
                f"SPLIT MANIFEST MISMATCH: The split manifest's training partition ('{training_split_key}') "
                f"has fingerprint {actual_train_fp}, which does not match the recorded train_split_fingerprint "
                f"({expected_train_fp}) in experiment metadata!"
            )

        # 2. Complete canonical split manifest fingerprint verification
        actual_manifest_fp = compute_split_manifest_fingerprint(split_data)
        expected_manifest_fp = config_metadata.get("split_manifest_fingerprint")
        if expected_manifest_fp and actual_manifest_fp != expected_manifest_fp:
            raise ValueError(
                f"SPLIT MANIFEST MISMATCH: The complete split manifest partition for '{training_split_key}' "
                f"has fingerprint {actual_manifest_fp}, which does not match the recorded split_manifest_fingerprint "
                f"({expected_manifest_fp}) in experiment metadata!"
            )

        # 3. Validation partition verification
        actual_val_imgs = split_data.get("val", [])
        actual_val_fp = hashlib.sha256(json.dumps(sorted(actual_val_imgs)).encode("utf-8")).hexdigest()
        expected_val_fp = config_metadata.get("val_split_fingerprint")
        if expected_val_fp and actual_val_fp != expected_val_fp:
            raise ValueError(
                f"SPLIT MANIFEST MISMATCH: The split manifest's validation partition ('{training_split_key}') "
                f"has fingerprint {actual_val_fp}, which does not match the recorded val_split_fingerprint "
                f"({expected_val_fp}) in experiment metadata!"
            )

        # 4. Test partition verification (for canonical split)
        if "test" in split_data or config_metadata.get("test_split_fingerprint"):
            actual_test_imgs = split_data.get("test", [])
            actual_test_fp = hashlib.sha256(json.dumps(sorted(actual_test_imgs)).encode("utf-8")).hexdigest()
            expected_test_fp = config_metadata.get("test_split_fingerprint")
            if expected_test_fp and actual_test_fp != expected_test_fp:
                raise ValueError(
                    f"SPLIT MANIFEST MISMATCH: The split manifest's test partition ('{training_split_key}') "
                    f"has fingerprint {actual_test_fp}, which does not match the recorded test_split_fingerprint "
                    f"({expected_test_fp}) in experiment metadata!"
                )

        # 5. Verify requested evaluation partition against experiment metadata
        if split_name is not None:
            norm_split = split_name.lower()
            if norm_split in ("canonical_val", "val"):
                if expected_val_fp and actual_val_fp != expected_val_fp:
                    raise ValueError(
                        f"EVALUATION PARTITION MISMATCH: Requested evaluation partition 'val' has fingerprint {actual_val_fp}, "
                        f"which does not match recorded val_split_fingerprint ({expected_val_fp})!"
                    )
            elif norm_split in ("canonical_test", "test", "canonical"):
                if expected_test_fp and actual_test_fp != expected_test_fp:
                    raise ValueError(
                        f"EVALUATION PARTITION MISMATCH: Requested evaluation partition 'test' has fingerprint {actual_test_fp}, "
                        f"which does not match recorded test_split_fingerprint ({expected_test_fp})!"
                    )

    return wt_path, tok_path, training_split_key, experiment_split_type, config_metadata

    return wt_path, tok_path, training_split_key, experiment_split_type, config_metadata


def evaluate_v2_bleu(
    model=None,
    cnn=None,
    weights_file=None,
    tokenizer_file=None,
    word_to_index=None,
    index_to_word=None,
    max_length=None,
    captions_dict=None,
    image_dir=None,
    target_images=None,
    split_name="historical_v2_val",
    trained_on=None,
    config_file=None,
    max_images=None
):
    """
    Evaluates V2 Attention model on unseen validation or test images using corpus BLEU-1 to BLEU-4.
    Uses streaming per-image generation to avoid allocating full feature arrays in memory.
    """
    # 1. Resolve and validate evaluation context, artifacts, and training split binding
    wt_path, tok_path, training_split_key, experiment_split_type, config_metadata = resolve_v2_evaluation_context(
        weights_file=weights_file,
        tokenizer_file=tokenizer_file,
        trained_on=trained_on,
        config_file=config_file,
        split_name=split_name
    )

    if word_to_index is None or index_to_word is None or max_length is None:
        word_to_index, index_to_word, max_length = load_tokenizer(tok_path)

    # Validate tokenizer vocabulary size against experiment metadata
    expected_vocab = config_metadata.get("vocab_size")
    if expected_vocab is not None and len(word_to_index) != expected_vocab:
        raise ValueError(
            f"TOKENIZER MISMATCH: Tokenizer vocabulary size ({len(word_to_index)}) does not match "
            f"expected vocab_size ({expected_vocab}) recorded in experiment metadata!"
        )

    if model is None:
        model = load_caption_model(
            vocab_size=len(word_to_index),
            weights_file=wt_path
        )

    if cnn is None:
        cnn = load_cnn()

    if captions_dict is None:
        # Load captions relative to REPO_ROOT/V1_Baseline_CNN_LSTM/data/captions/captions.txt
        captions_file = REPO_ROOT / "V1_Baseline_CNN_LSTM" / "data" / "captions" / "captions.txt"
        with open(captions_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        captions_dict = {}
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            image_name, caption = line.split(",", 1)
            captions_dict.setdefault(image_name, []).append(caption)
        captions_dict = clean_all_captions(captions_dict)

    if image_dir is None:
        image_dir = DEFAULT_IMAGE_DIR

    image_dir = Path(image_dir).resolve()

    # 2. Resolve target evaluation images and enforce leak guard against validated training split
    if target_images is None:
        if SPLITS_FILE.exists():
            with open(SPLITS_FILE, "r", encoding="utf-8") as f:
                splits = json.load(f)

            trained_images_set = set(splits.get(training_split_key, {}).get("train", []))

            # Resolve target evaluation images from requested split
            if split_name in ("historical_v2_val", "historical"):
                target_images = splits.get("historical_v2_split", {}).get("val", [])
            elif split_name in ("canonical_val", "val"):
                target_images = splits.get("canonical_split", {}).get("val", [])
            elif split_name in ("canonical_test", "test", "canonical"):
                target_images = splits.get("canonical_split", {}).get("test", [])
            elif split_name in ("historical_v1_val",):
                target_images = splits.get("historical_v1_split", {}).get("val", [])
            else:
                target_images = splits.get(split_name, [])

            # Rigorous leak check: Compare evaluated images directly against the validated training set
            if trained_images_set:
                overlap = set(target_images).intersection(trained_images_set)
                if overlap:
                    raise ValueError(
                        f"DATA LEAK DETECTED: {len(overlap)} evaluation images in split '{split_name}' "
                        f"were found in the training split ('{training_split_key}') associated with checkpoint '{wt_path.name}'! "
                        f"Historical checkpoints must only be evaluated on their matching historical validation split."
                    )
        else:
            # Fallback to deterministic reproduction of V2 validation split
            all_keys = list(captions_dict.keys())
            import random
            random.seed(42)
            random.shuffle(all_keys)
            split_idx = int(len(all_keys) * 0.10)
            target_images = all_keys[:split_idx]

    if max_images is not None and max_images > 0:
        target_images = target_images[:max_images]

    actual = []
    predicted = []
    evaluated_images = []

    total_target = len(target_images)
    print(f"\n========================================================")
    print(f"V2 BLEU Evaluation | Split: {split_name} | Images: {total_target}")
    print(f"========================================================\n")

    for i, image_name in enumerate(target_images):
        image_path = image_dir / image_name
        if not image_path.exists() or image_name not in captions_dict:
            continue

        # Streaming evaluation: process one image at a time, no massive memory allocation
        prediction = generate_caption(
            image_path=image_path,
            model=model,
            cnn=cnn,
            word_to_index=word_to_index,
            index_to_word=index_to_word,
            max_length=max_length
        )

        pred_tokens = [w for w in prediction.split() if w not in {"<start>", "<end>", "<pad>"}]
        predicted.append(pred_tokens)

        references = []
        for caption in captions_dict[image_name]:
            ref_clean = caption.replace("<start>", "").replace("<end>", "").strip()
            ref_tokens = ref_clean.split()
            references.append(ref_tokens)

        actual.append(references)
        evaluated_images.append(image_name)

        if (i + 1) % 25 == 0 or (i + 1) == total_target:
            print(f"Evaluated {i + 1}/{total_target} images...")

    if _HAS_NLTK:
        smooth = SmoothingFunction().method1
        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
        bleu3 = corpus_bleu(actual, predicted, weights=(1/3, 1/3, 1/3, 0.0), smoothing_function=smooth)
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    else:
        bleu1 = compute_corpus_bleu(actual, predicted, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2 = compute_corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3 = compute_corpus_bleu(actual, predicted, weights=(1/3, 1/3, 1/3, 0.0))
        bleu4 = compute_corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    results = {
        "model_version": "V2_Attention",
        "split_used": split_name,
        "images_evaluated": len(evaluated_images),
        "total_references": sum(len(r) for r in actual),
        "bleu1": float(bleu1),
        "bleu2": float(bleu2),
        "bleu3": float(bleu3),
        "bleu4": float(bleu4)
    }

    print("\n==============================")
    print("V2 HELD-OUT BLEU SCORES")
    print("==============================")
    print(f"Split Evaluated : {split_name}")
    print(f"Images Evaluated: {len(evaluated_images)}")
    print(f"BLEU-1          : {bleu1:.4f}")
    print(f"BLEU-2          : {bleu2:.4f}")
    print(f"BLEU-3          : {bleu3:.4f}")
    print(f"BLEU-4          : {bleu4:.4f}")
    print("==============================\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V2 Attention model using corpus BLEU.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to evaluate (default: all)")
    parser.add_argument("--split", type=str, default="historical_v2_val", choices=["historical_v2_val", "canonical_val", "canonical_test"], help="Evaluation split name (default: historical_v2_val)")
    parser.add_argument("--trained_on", type=str, default=None, choices=["historical_v2", "canonical"], help="Training split of the evaluated model (default: auto-detected from metadata)")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.weights.h5)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config.json (for non-historical models)")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to images directory")
    args = parser.parse_args()

    evaluate_v2_bleu(
        weights_file=args.weights,
        tokenizer_file=args.tokenizer,
        image_dir=args.image_dir,
        config_file=args.config,
        max_images=args.max_images,
        split_name=args.split,
        trained_on=args.trained_on
    )
