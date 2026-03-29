import argparse
import math
import os
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


def predict_match(similarities: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """
    Converts similarity scores into binary predictions.
    1 = matched / consistent, 0 = mismatched / inconsistent
    """

    return (similarities >= threshold).astype(int)


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
    Saves a distribution plot for similarity scores of positive and negative samples.
    """

    plt.figure(figsize=(8, 5))
    positive_scores = results_df.loc[results_df["label"] == 1, "similarity"]
    negative_scores = results_df.loc[results_df["label"] == 0, "similarity"]

    if len(positive_scores) > 0:
        sns.histplot(positive_scores, bins=30, kde=True, color="green", label="Matched (label=1)", alpha=0.5)
    if len(negative_scores) > 0:
        sns.histplot(negative_scores, bins=30, kde=True, color="red", label="Mismatched (label=0)", alpha=0.5)

    plt.title("Similarity Score Distribution")
    plt.xlabel("Cosine Similarity")
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
            f"Similarity: {row['similarity']:.4f}"
        )

        axis.set_title(title, fontsize=11)
        wrapped_text = textwrap.fill(f"Text: {row['text']}", width=45)
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
    threshold: float,
    model_name: str,
    pretrained: str,
    num_visual_samples: int,
) -> None:
    """
    End-to-end inference pipeline:
    1. Load model and dataset
    2. Encode image-text pairs
    3. Compute cosine similarity
    4. Predict labels using a threshold
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
        text_features = encode_text(model, tokenizer, batch["texts"], device)
        similarities = compute_similarity(image_features, text_features).cpu().numpy()
        predictions = predict_match(similarities, threshold=threshold)
        labels = batch["labels"].cpu().numpy()

        for image_path, text, label, similarity, prediction in zip(
            batch["image_paths"],
            batch["texts"],
            labels,
            similarities,
            predictions,
        ):
            all_results.append(
                {
                    "image_path": image_path,
                    "text": text,
                    "label": int(label),
                    "similarity": float(similarity),
                    "prediction": int(prediction),
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
    print(f"[INFO] Threshold used for classification: {threshold}")


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
        "--threshold",
        type=float,
        default=0.25,
        help="Similarity threshold for binary classification.",
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
        threshold=args.threshold,
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_visual_samples=args.num_visual_samples,
    )
