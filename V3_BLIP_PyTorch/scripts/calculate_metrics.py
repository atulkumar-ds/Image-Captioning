
"""Compare pretrained BLIP and Flickr30k-finetuned BLIP.

Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- Average inference time

Inputs:
    results/evaluation/test_predictions_baseline.csv
    results/evaluation/test_predictions_finetuned.csv

Outputs:
    results/evaluation/metrics_comparison.json
    results/evaluation/metrics_comparison.csv
"""

import csv
import json
import re
import zipfile
from pathlib import Path

import nltk
from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction,
)
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "evaluation"

BASELINE_CSV = RESULTS_DIR / "test_predictions_baseline.csv"
FINETUNED_CSV = RESULTS_DIR / "test_predictions_finetuned.csv"


def ensure_wordnet():
    """Make NLTK WordNet available, including when downloaded as a ZIP."""
    nltk_dir = Path.home() / "nltk_data"
    corpus_dir = nltk_dir / "corpora"
    wordnet_dir = corpus_dir / "wordnet"
    wordnet_zip = corpus_dir / "wordnet.zip"

    nltk_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Prioritize the user's standard NLTK data directory.
    nltk_path = str(nltk_dir)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)

    # Download WordNet only when its files are not already available.
    if not (wordnet_dir / "index.noun").is_file():
        if not wordnet_zip.is_file():
            print("Downloading NLTK WordNet resource...")
            success = nltk.download(
                "wordnet",
                download_dir=nltk_path,
                quiet=False,
            )
            if not success:
                raise RuntimeError(
                    "WordNet download failed. Check your internet connection."
                )

        # Some installations retain WordNet as wordnet.zip.
        # Extract it into corpora so NLTK can find corpora/wordnet.
        if wordnet_zip.is_file():
            print(f"Extracting WordNet archive: {wordnet_zip}")
            with zipfile.ZipFile(wordnet_zip, "r") as archive:
                archive.extractall(corpus_dir)

    if not (wordnet_dir / "index.noun").is_file():
        raise RuntimeError(
            "WordNet files were not found after installation. "
            f"Expected: {wordnet_dir / 'index.noun'}"
        )

    # Verify the actual corpus loader, not just the directory path.
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    wordnet.synsets("dog")

    print(f"WordNet is ready: {wordnet_dir}")


def tokenize(text):
    """Normalize a caption into lowercase word and punctuation tokens."""
    return re.findall(
        r"[a-z0-9]+(?:['’][a-z0-9]+)?|[^\w\s]",
        text.lower(),
    )


def load_predictions(csv_path):
    """Load generated captions and five reference captions per image."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {csv_path}"
        )

    predictions = {}

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"image", "caption", "references"}
        available_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(available_columns):
            raise ValueError(
                f"Unexpected columns in {csv_path.name}. "
                f"Expected {required_columns}; "
                f"found {available_columns}."
            )

        for row in reader:
            image_name = Path(row["image"]).name
            caption = (row["caption"] or "").strip()

            if image_name in predictions:
                raise ValueError(
                    f"Duplicate prediction for image: {image_name}"
                )

            if not caption:
                raise ValueError(
                    f"Empty generated caption for: {image_name}"
                )

            try:
                references = json.loads(row["references"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid reference-caption JSON for {image_name}"
                ) from exc

            if not isinstance(references, list) or len(references) != 5:
                raise ValueError(
                    f"Expected five reference captions for {image_name}"
                )

            if any(
                not isinstance(ref, str) or not ref.strip()
                for ref in references
            ):
                raise ValueError(
                    f"Empty or invalid reference caption for {image_name}"
                )

            try:
                inference_seconds = float(row["inference_seconds"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid inference time for {image_name}"
                ) from exc

            predictions[image_name] = {
                "caption": caption,
                "references": references,
                "inference_seconds": inference_seconds,
            }

    if not predictions:
        raise ValueError(f"No predictions found in {csv_path}")

    return predictions


def calculate_metrics(predictions):
    """Calculate corpus-level and per-image caption-quality metrics."""
    image_names = sorted(predictions)

    references_text = [
        predictions[name]["references"]
        for name in image_names
    ]
    hypotheses_text = [
        predictions[name]["caption"]
        for name in image_names
    ]

    references_tokens = [
        [tokenize(reference) for reference in references]
        for references in references_text
    ]
    hypotheses_tokens = [
        tokenize(hypothesis)
        for hypothesis in hypotheses_text
    ]

    smoothing = SmoothingFunction().method1
    metrics = {}

    # Corpus BLEU scores are reported on the conventional 0-100 scale.
    for n in range(1, 5):
        weights = tuple(
            [1.0 / n] * n + [0.0] * (4 - n)
        )

        score = corpus_bleu(
            references_tokens,
            hypotheses_tokens,
            weights=weights,
            smoothing_function=smoothing,
        )
        metrics[f"BLEU-{n}"] = score * 100.0

    # METEOR uses tokenized captions and WordNet-based matching.
    meteor_values = [
        meteor_score(references, hypothesis)
        for references, hypothesis in zip(
            references_tokens,
            hypotheses_tokens,
        )
    ]
    metrics["METEOR"] = (
        sum(meteor_values) / len(meteor_values) * 100.0
    )

    # ROUGE-L: use the best matching reference for each image.
    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    rouge_values = []
    for references, hypothesis in zip(
        references_text,
        hypotheses_text,
    ):
        best_score = max(
            rouge.score(reference, hypothesis)["rougeL"].fmeasure
            for reference in references
        )
        rouge_values.append(best_score)

    metrics["ROUGE-L"] = (
        sum(rouge_values) / len(rouge_values) * 100.0
    )

    # CIDEr uses the pycocoevalcap implementation.
    # Whitespace-separated normalized tokens are passed to its scorer.
    cider_gts = {}
    cider_res = {}

    for index, (references, hypothesis) in enumerate(
        zip(references_text, hypotheses_text)
    ):
        cider_gts[index] = [
            " ".join(tokenize(reference))
            for reference in references
        ]
        cider_res[index] = [
            " ".join(tokenize(hypothesis))
        ]

    cider_score, _ = Cider().compute_score(
        cider_gts,
        cider_res,
    )
    metrics["CIDEr"] = float(cider_score)

    metrics["Average inference seconds"] = sum(
        predictions[name]["inference_seconds"]
        for name in image_names
    ) / len(image_names)

    metrics["Number of images"] = len(image_names)

    return metrics


def print_comparison(baseline_metrics, finetuned_metrics):
    """Print both models' scores in a readable table."""
    print("\n" + "=" * 76)
    print("IMAGE CAPTIONING EVALUATION")
    print("=" * 76)

    print(
        f"{'Metric':<30}"
        f"{'Original BLIP':>20}"
        f"{'Fine-tuned BLIP':>24}"
    )
    print("-" * 76)

    for metric_name in baseline_metrics:
        baseline_value = baseline_metrics[metric_name]
        finetuned_value = finetuned_metrics[metric_name]

        if metric_name == "Number of images":
            print(
                f"{metric_name:<30}"
                f"{baseline_value:>20.0f}"
                f"{finetuned_value:>24.0f}"
            )
        elif metric_name == "Average inference seconds":
            print(
                f"{metric_name:<30}"
                f"{baseline_value:>20.4f}"
                f"{finetuned_value:>24.4f}"
            )
        else:
            print(
                f"{metric_name:<30}"
                f"{baseline_value:>20.4f}"
                f"{finetuned_value:>24.4f}"
            )

    print("=" * 76)

    print("\nMetric interpretation:")
    print("- Higher BLEU, METEOR, ROUGE-L and CIDEr generally indicate")
    print("  stronger agreement with the reference captions.")
    print("- Lower inference time means faster caption generation.")
    print("- These metrics alone do not establish factual correctness.")


def save_results(baseline_metrics, finetuned_metrics, image_count):
    """Save the evaluation report as JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "Flickr30k test split",
        "number_of_images": image_count,
        "reference_captions_per_image": 5,
        "baseline_model": "Salesforce/blip-image-captioning-base",
        "fine_tuned_model": "checkpoints/flickr30k_blip",
        "metric_notes": {
            "BLEU": "NLTK corpus BLEU with method1 smoothing, 0-100 scale.",
            "METEOR": "NLTK METEOR with WordNet.",
            "ROUGE-L": "Mean of the best reference ROUGE-L F1 per image.",
            "CIDEr": (
                "pycocoevalcap Cider on whitespace-separated normalized "
                "tokens; record this tokenization when comparing scores."
            ),
            "Inference time": "Average from the saved prediction CSV files.",
        },
        "baseline_metrics": baseline_metrics,
        "fine_tuned_metrics": finetuned_metrics,
    }

    json_path = RESULTS_DIR / "metrics_comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    csv_path = RESULTS_DIR / "metrics_comparison.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "metric",
            "original_blip",
            "flickr30k_finetuned",
        ])

        for metric_name in baseline_metrics:
            writer.writerow([
                metric_name,
                baseline_metrics[metric_name],
                finetuned_metrics[metric_name],
            ])

    print(f"\nJSON report saved to: {json_path}")
    print(f"CSV report saved to:  {csv_path}")


def main():
    ensure_wordnet()

    print("\nLoading baseline predictions...")
    baseline = load_predictions(BASELINE_CSV)

    print("Loading fine-tuned predictions...")
    finetuned = load_predictions(FINETUNED_CSV)

    if set(baseline) != set(finetuned):
        raise ValueError(
            "Baseline and fine-tuned predictions do not contain "
            "the same images."
        )

    for image_name in baseline:
        if (
            baseline[image_name]["references"]
            != finetuned[image_name]["references"]
        ):
            raise ValueError(
                f"Reference captions differ for {image_name}."
            )

    image_count = len(baseline)
    print(f"\nImages evaluated: {image_count}")
    print("Reference captions per image: 5")
    print("Computing metrics...")

    print("\nEvaluating original BLIP...")
    baseline_metrics = calculate_metrics(baseline)

    print("Evaluating Flickr30k fine-tuned BLIP...")
    finetuned_metrics = calculate_metrics(finetuned)

    print_comparison(baseline_metrics, finetuned_metrics)

    save_results(
        baseline_metrics,
        finetuned_metrics,
        image_count,
    )


if __name__ == "__main__":
    main()
