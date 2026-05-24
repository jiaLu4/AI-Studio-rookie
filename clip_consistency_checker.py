import argparse
import math
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PRODUCTNESS_PROMPTS = [
    "a clean e-commerce product photo",
    "a product listing image with one main product",
    "a retail product image on a simple background",
    "a clear product-focused catalog photo",
]

INVALID_IMAGE_PROMPTS = [
    "a screenshot of a shopping app or webpage",
    "a promotional poster with large text",
    "a collage of multiple images",
    "a selfie or unrelated person photo",
    "a cluttered scene with no clear main product",
    "a marketing banner or advertisement",
]


@dataclass
class SampleRecord:
    image_path: str
    text: str
    label: int


class ImageTextDataset(Dataset):
    """
    Simple dataset for loading image-text-label rows from a CSV file.
    Invalid images are handled safely and skipped later in the collate function.
    """

    def __init__(self, dataframe: pd.DataFrame, preprocess):
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict:
        row = self.dataframe.iloc[index]
        image_path = str(row["image_path"])
        text = str(row["text"])
        label = int(row["label"])

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image)
            is_valid = True
            error_message = None
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            image_tensor = None
            is_valid = False
            error_message = str(error)

        return {
            "image_path": image_path,
            "text": text,
            "label": label,
            "image_tensor": image_tensor,
            "is_valid": is_valid,
            "error_message": error_message,
        }


def normalize_text(text: str) -> str:
    """
    Normalizes raw metadata text into a CLIP-friendlier sentence fragment.
    """

    text = str(text).replace("|", ", ")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def parse_structured_text(text: str) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Parses the pipe-separated product text used in the Stage 1 CSV.
    Returns a subject string, lightweight attributes, and extra descriptors.
    """

    parts = [part.strip() for part in str(text).split("|") if part.strip()]
    subject_parts: List[str] = []
    attributes: Dict[str, str] = {}
    extras: List[str] = []

    for part in parts:
        lowered = part.lower()
        if lowered.startswith("color:"):
            attributes["color"] = part.split(":", 1)[1].strip()
        elif lowered.startswith("style:"):
            attributes["style"] = part.split(":", 1)[1].strip()
        elif lowered.startswith("type:"):
            attributes["type"] = part.split(":", 1)[1].strip()
        elif len(subject_parts) < 2:
            subject_parts.append(part)
        else:
            extras.append(part)

    subject = ", ".join(subject_parts).strip(", ")
    if not subject:
        subject = normalize_text(text)

    return subject, attributes, extras


def build_description_prompts(text: str) -> List[str]:
    """
    Creates a small prompt ensemble for the same product description.
    This is more stable than relying on a single raw metadata string.
    """

    normalized_text = normalize_text(text)
    subject, attributes, extras = parse_structured_text(text)

    attribute_fragments: List[str] = []
    if attributes.get("type"):
        attribute_fragments.append(f"type {attributes['type']}")
    if attributes.get("color"):
        attribute_fragments.append(f"color {attributes['color']}")
    if attributes.get("style"):
        attribute_fragments.append(f"style {attributes['style']}")

    attribute_suffix = ""
    if attribute_fragments:
        attribute_suffix = ", " + ", ".join(attribute_fragments)

    extra_fragment = ""
    if extras:
        extra_fragment = f", details: {extras[0]}"

    prompt_candidates = [
        normalized_text,
        f"a product listing image of {subject}",
        f"a clear e-commerce product photo of {subject}{attribute_suffix}",
        f"the main subject is {subject}{attribute_suffix}{extra_fragment}",
    ]

    unique_prompts = list(dict.fromkeys(prompt.strip(" .") + "." for prompt in prompt_candidates if prompt.strip()))
    while len(unique_prompts) < 4:
        unique_prompts.append(unique_prompts[-1])
    return unique_prompts[:4]


def build_prompt_matrix(texts: List[str]) -> List[List[str]]:
    return [build_description_prompts(text) for text in texts]


def collate_valid_samples(batch: List[Dict]) -> Dict:
    """
    Filters out invalid images while keeping track of skipped rows.
    """

    valid_samples = [item for item in batch if item["is_valid"]]
    invalid_samples = [item for item in batch if not item["is_valid"]]

    if valid_samples:
        image_tensors = torch.stack([item["image_tensor"] for item in valid_samples])
        texts = [item["text"] for item in valid_samples]
        labels = torch.tensor([item["label"] for item in valid_samples], dtype=torch.long)
        image_paths = [item["image_path"] for item in valid_samples]
    else:
        image_tensors = None
        texts = []
        labels = torch.tensor([], dtype=torch.long)
        image_paths = []

    return {
        "image_tensors": image_tensors,
        "texts": texts,
        "labels": labels,
        "image_paths": image_paths,
        "invalid_samples": invalid_samples,
    }


def load_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> Tuple[torch.nn.Module, object, object, torch.device]:
    """
    Loads OpenCLIP model, preprocessing transforms, tokenizer, and device.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loaded model: {model_name}")
    print(f"[INFO] Loaded pretrained weights: {pretrained}")

    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_image(model: torch.nn.Module, image_tensors: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encodes a batch of images into normalized CLIP embeddings.
    """

    image_tensors = image_tensors.to(device)
    image_features = model.encode_image(image_tensors)
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features


@torch.no_grad()
def encode_text(model: torch.nn.Module, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    Encodes a batch of texts into normalized CLIP embeddings.
    """

    tokenized_texts = tokenizer(texts).to(device)
    text_features = model.encode_text(tokenized_texts)
    text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features


def compute_similarity(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between matching image-text pairs.
    Since features are normalized, cosine similarity is a dot product.
    """

    similarities = torch.sum(image_features * text_features, dim=-1)
    return similarities


def encode_text_prompt_matrix(
    model: torch.nn.Module,
    tokenizer,
    prompt_matrix: List[List[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    Encodes a fixed prompt ensemble for each sample.
    Returns a tensor of shape [batch_size, prompts_per_sample, embedding_dim].
    """

    prompts_per_sample = len(prompt_matrix[0])
    flat_prompts = [prompt for prompts in prompt_matrix for prompt in prompts]
    flat_features = encode_text(model, tokenizer, flat_prompts, device)
    embedding_dim = flat_features.shape[-1]
    return flat_features.view(len(prompt_matrix), prompts_per_sample, embedding_dim)


def precompute_prompt_bank_features(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    return encode_text(model, tokenizer, prompts, device)


def compute_prompt_ensemble_score(
    image_features: torch.Tensor,
    prompt_features: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the mean similarity against a per-sample prompt ensemble.
    prompt_features shape: [batch_size, num_prompts, embedding_dim]
    """

    return torch.einsum("bd,bpd->bp", image_features, prompt_features).mean(dim=1)


def compute_prompt_bank_score(
    image_features: torch.Tensor,
    prompt_bank_features: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the max similarity against a global prompt bank.
    prompt_bank_features shape: [num_prompts, embedding_dim]
    """

    return image_features @ prompt_bank_features.T


def combine_scores(
    match_scores: np.ndarray,
    product_scores: np.ndarray,
    invalid_scores: np.ndarray,
    match_weight: float,
    product_weight: float,
    invalid_weight: float,
) -> np.ndarray:
    return (
        match_weight * match_scores
        + product_weight * product_scores
        - invalid_weight * invalid_scores
    )


def predict_match(
    final_scores: np.ndarray,
    product_margins: np.ndarray,
    match_margins: np.ndarray,
    invalid_scores: np.ndarray,
    positive_threshold: float,
    negative_threshold: float,
    product_margin_threshold: float,
    invalid_score_threshold: float,
    uncertain_as_positive: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts enhanced CLIP scores into:
    - binary predictions for metrics
    - a 3-way decision label
    - a short explanation reason
    """

    positive_mask = (
        (final_scores >= positive_threshold)
        & (product_margins >= product_margin_threshold)
        & (invalid_scores <= invalid_score_threshold)
    )

    negative_mask = (
        (final_scores <= negative_threshold)
        | (invalid_scores >= invalid_score_threshold + 0.05)
        | ((product_margins < 0.0) & (match_margins < 0.0))
    )

    decisions = np.full(len(final_scores), "uncertain", dtype=object)
    decisions[positive_mask] = "consistent"
    decisions[negative_mask & ~positive_mask] = "inconsistent"

    predictions = np.zeros(len(final_scores), dtype=int)
    predictions[decisions == "consistent"] = 1
    if uncertain_as_positive:
        predictions[decisions == "uncertain"] = 1

    reasons = np.empty(len(final_scores), dtype=object)
    for index in range(len(final_scores)):
        if decisions[index] == "consistent":
            reasons[index] = "strong product-image and text alignment"
        elif invalid_scores[index] >= invalid_score_threshold + 0.05:
            reasons[index] = "image is highly similar to screenshot/poster/clutter prompts"
        elif product_margins[index] < 0.0 and match_margins[index] < 0.0:
            reasons[index] = "product evidence is weaker than invalid-image evidence"
        elif final_scores[index] <= negative_threshold:
            reasons[index] = "weak product/text consistency score"
        else:
            reasons[index] = "borderline case; send to manual review"

    return predictions, decisions, reasons


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Computes standard binary classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1_score": f1_score(labels, predictions, zero_division=0),
    }
    return metrics


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: str) -> None:
    """
    Saves a confusion matrix plot.
    """

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Inconsistent", "Pred: Consistent"],
        yticklabels=["True: Inconsistent", "True: Consistent"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved confusion matrix to: {output_path}")


def plot_similarity_distribution(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves a distribution plot for the final Stage 1 decision score.
    """

    plt.figure(figsize=(8, 5))
    positive_scores = results_df.loc[results_df["label"] == 1, "final_score"]
    negative_scores = results_df.loc[results_df["label"] == 0, "final_score"]

    if len(positive_scores) > 0:
        sns.histplot(positive_scores, bins=30, kde=True, color="green", label="Matched (label=1)", alpha=0.5)
    if len(negative_scores) > 0:
        sns.histplot(negative_scores, bins=30, kde=True, color="red", label="Mismatched (label=0)", alpha=0.5)

    plt.title("Final Decision Score Distribution")
    plt.xlabel("Final Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved similarity distribution plot to: {output_path}")


def visualize_samples(
    results_df: pd.DataFrame,
    output_path: str,
    num_samples: int = 8,
    random_seed: int = 42,
) -> None:
    """
    Visualizes sample predictions with image, text, labels, and similarity scores.
    """

    if results_df.empty:
        print("[WARN] No valid predictions available for sample visualization.")
        return

    sample_count = min(num_samples, len(results_df))
    sampled_df = results_df.sample(sample_count, random_state=random_seed).reset_index(drop=True)

    columns = 2
    rows = math.ceil(sample_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 7 * rows))
    axes = np.array(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        axis = axes[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            axis.imshow(image)
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            axis.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=12)

        true_name = "Consistent" if int(row["label"]) == 1 else "Inconsistent"
        pred_name = "Consistent" if int(row["prediction"]) == 1 else "Inconsistent"

        title = (
            f"True: {true_name}\n"
            f"Pred: {pred_name}\n"
            f"Decision: {row['decision']}\n"
            f"Final score: {row['final_score']:.4f}"
        )

        axis.set_title(title, fontsize=11)
        wrapped_text = textwrap.fill(
            f"Text: {row['text']}\nReason: {row['reason']}\n"
            f"match={row['match_score']:.3f}, product={row['product_score']:.3f}, invalid={row['invalid_score']:.3f}",
            width=45,
        )
        axis.text(
            0.5,
            -0.12,
            wrapped_text,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc"),
        )
        axis.axis("off")

    plt.tight_layout(h_pad=4.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved sample visualization to: {output_path}")


def run_inference(
    csv_path: str,
    output_dir: str,
    batch_size: int,
    positive_threshold: float,
    negative_threshold: float,
    product_margin_threshold: float,
    invalid_score_threshold: float,
    match_weight: float,
    product_weight: float,
    invalid_weight: float,
    uncertain_as_positive: bool,
    model_name: str,
    pretrained: str,
    num_visual_samples: int,
) -> None:
    """
    End-to-end inference pipeline:
    1. Load model and dataset
    2. Encode image-text pairs
    3. Compute description match + productness + invalid-image prototype scores
    4. Combine them into a margin-based decision score
    5. Predict labels using dual thresholds
    5. Evaluate and save outputs
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    dataframe = pd.read_csv(csv_path)
    required_columns = {"image_path", "text", "label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

    dataframe = dataframe.dropna(subset=["image_path", "text", "label"]).copy()
    dataframe["image_path"] = dataframe["image_path"].astype(str)
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    print(f"[INFO] Loaded {len(dataframe)} rows from: {csv_path}")

    model, preprocess, tokenizer, device = load_model(model_name=model_name, pretrained=pretrained)
    product_prompt_features = precompute_prompt_bank_features(
        model,
        tokenizer,
        PRODUCTNESS_PROMPTS,
        device,
    )
    invalid_prompt_features = precompute_prompt_bank_features(
        model,
        tokenizer,
        INVALID_IMAGE_PROMPTS,
        device,
    )
    dataset = ImageTextDataset(dataframe, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_valid_samples,
    )

    all_results: List[Dict] = []
    invalid_records: List[Dict] = []

    for batch in tqdm(dataloader, desc="Running inference"):
        invalid_records.extend(batch["invalid_samples"])

        if batch["image_tensors"] is None:
            continue

        image_features = encode_image(model, batch["image_tensors"], device)
        prompt_matrix = build_prompt_matrix(batch["texts"])
        prompt_features = encode_text_prompt_matrix(model, tokenizer, prompt_matrix, device)

        match_scores = compute_prompt_ensemble_score(image_features, prompt_features).cpu().numpy()
        product_scores = compute_prompt_bank_score(image_features, product_prompt_features).max(dim=1).values.cpu().numpy()
        invalid_scores = compute_prompt_bank_score(image_features, invalid_prompt_features).max(dim=1).values.cpu().numpy()
        product_margins = product_scores - invalid_scores
        match_margins = match_scores - invalid_scores
        final_scores = combine_scores(
            match_scores=match_scores,
            product_scores=product_scores,
            invalid_scores=invalid_scores,
            match_weight=match_weight,
            product_weight=product_weight,
            invalid_weight=invalid_weight,
        )
        predictions, decisions, reasons = predict_match(
            final_scores=final_scores,
            product_margins=product_margins,
            match_margins=match_margins,
            invalid_scores=invalid_scores,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            product_margin_threshold=product_margin_threshold,
            invalid_score_threshold=invalid_score_threshold,
            uncertain_as_positive=uncertain_as_positive,
        )
        labels = batch["labels"].cpu().numpy()

        for image_path, text, label, match_score, product_score, invalid_score, product_margin, match_margin, final_score, prediction, decision, reason in zip(
            batch["image_paths"],
            batch["texts"],
            labels,
            match_scores,
            product_scores,
            invalid_scores,
            product_margins,
            match_margins,
            final_scores,
            predictions,
            decisions,
            reasons,
        ):
            all_results.append(
                {
                    "image_path": image_path,
                    "text": text,
                    "label": int(label),
                    "similarity": float(match_score),
                    "match_score": float(match_score),
                    "product_score": float(product_score),
                    "invalid_score": float(invalid_score),
                    "product_margin": float(product_margin),
                    "match_margin": float(match_margin),
                    "final_score": float(final_score),
                    "prediction": int(prediction),
                    "decision": str(decision),
                    "reason": str(reason),
                }
            )

    if invalid_records:
        print(f"[WARN] Skipped {len(invalid_records)} invalid image(s).")
        for sample in invalid_records[:5]:
            print(f"       - {sample['image_path']} | Error: {sample['error_message']}")
        if len(invalid_records) > 5:
            print("       - Additional invalid images omitted from log.")

    if not all_results:
        raise RuntimeError("No valid image-text pairs were processed. Please check your dataset and image paths.")

    results_df = pd.DataFrame(all_results)
    labels = results_df["label"].to_numpy()
    predictions = results_df["prediction"].to_numpy()

    metrics = evaluate_predictions(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, digits=4, zero_division=0)

    print("\n[RESULTS] Evaluation Metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    uncertain_count = int((results_df["decision"] == "uncertain").sum())
    print(f"Uncertain: {uncertain_count}")
    print("\n[RESULTS] Confusion Matrix")
    print(cm)
    print("\n[RESULTS] Classification Report")
    print(report)

    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    similarity_distribution_path = os.path.join(output_dir, "similarity_distribution.png")
    sample_visualization_path = os.path.join(output_dir, "sample_predictions.png")

    results_df.to_csv(predictions_csv_path, index=False)
    print(f"[INFO] Saved predictions to: {predictions_csv_path}")

    plot_confusion_matrix(labels, predictions, confusion_matrix_path)
    plot_similarity_distribution(results_df, similarity_distribution_path)
    visualize_samples(results_df, sample_visualization_path, num_samples=num_visual_samples)

    print("\n[INFO] Stage 1 prototype completed successfully.")
    print(f"[INFO] Positive threshold used: {positive_threshold}")
    print(f"[INFO] Negative threshold used: {negative_threshold}")
    print(f"[INFO] Product-margin threshold used: {product_margin_threshold}")
    print(f"[INFO] Invalid-score threshold used: {invalid_score_threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP-based Zero-shot / Weakly-supervised Multimodal Consistency Checker"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV dataset file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where predictions and plots will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.30,
        help="High-confidence threshold for accepting a product/image match.",
    )
    parser.add_argument(
        "--negative_threshold",
        type=float,
        default=0.18,
        help="Low-confidence threshold for rejecting a product/image match.",
    )
    parser.add_argument(
        "--product_margin_threshold",
        type=float,
        default=0.02,
        help="Minimum product-vs-invalid margin required for a positive decision.",
    )
    parser.add_argument(
        "--invalid_score_threshold",
        type=float,
        default=0.26,
        help="Maximum invalid-image prototype similarity allowed for a positive decision.",
    )
    parser.add_argument(
        "--match_weight",
        type=float,
        default=0.75,
        help="Weight for description-to-image alignment score.",
    )
    parser.add_argument(
        "--product_weight",
        type=float,
        default=0.20,
        help="Weight for product-image prototype similarity.",
    )
    parser.add_argument(
        "--invalid_weight",
        type=float,
        default=0.50,
        help="Penalty weight for invalid-image prototype similarity.",
    )
    parser.add_argument(
        "--uncertain_as_positive",
        action="store_true",
        help="Map uncertain cases to label 1 when computing binary metrics. Default keeps uncertain as negative/review.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model backbone.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights name.",
    )
    parser.add_argument(
        "--num_visual_samples",
        type=int,
        default=8,
        help="Number of prediction samples to visualize.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        product_margin_threshold=args.product_margin_threshold,
        invalid_score_threshold=args.invalid_score_threshold,
        match_weight=args.match_weight,
        product_weight=args.product_weight,
        invalid_weight=args.invalid_weight,
        uncertain_as_positive=args.uncertain_as_positive,
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_visual_samples=args.num_visual_samples,
    )
