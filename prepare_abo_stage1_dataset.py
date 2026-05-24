import argparse
import ast
import gzip
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


NON_LATIN_PATTERN = re.compile(
    r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F"
    r"\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an ABO Stage 1 image-text consistency dataset."
    )
    parser.add_argument(
        "--abo_root",
        type=str,
        required=True,
        help="Path to the extracted ABO root directory containing listings/ and images/.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="abo_stage1_pairs.csv",
        help="Path to save the generated CSV.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=1000,
        help="Maximum number of products to use when building the dataset.",
    )
    parser.add_argument(
        "--negatives_per_positive",
        type=int,
        default=1,
        help="How many negative texts to sample per positive pair.",
    )
    parser.add_argument(
        "--language_priority",
        type=str,
        default="en_US,en_GB,en_IN,en_AU,en_SG",
        help="Comma-separated language priority list for selecting text fields.",
    )
    parser.add_argument(
        "--prefer_main_image",
        action="store_true",
        help="Use only the main image for each product when available.",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=8,
        help="Minimum text length required to keep a sample.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def safe_literal(value):
    """
    ABO metadata fields may contain Python-like serialized dict strings when read from
    some sources. This helper turns them into Python objects when needed.
    """

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("{") or value.startswith("["):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value


def ensure_list(value) -> List:
    if value is None:
        return []
    value = safe_literal(value)
    if isinstance(value, list):
        return value
    return [value]


def load_jsonl_gz(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_image_index(images_metadata_path: Path) -> Dict[str, str]:
    """
    Maps ABO image_id -> relative path like 14/14fe8812.jpg.
    """

    if not images_metadata_path.exists():
        raise FileNotFoundError(f"Image metadata file not found: {images_metadata_path}")

    df = pd.read_csv(images_metadata_path, compression="gzip")
    required_columns = {"image_id", "path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in images metadata: {sorted(missing_columns)}")

    return dict(zip(df["image_id"].astype(str), df["path"].astype(str)))


def is_english_like(text: Optional[str]) -> bool:
    """
    Keep only text that looks like Latin-script product metadata.
    This helps avoid mixed-language samples in Stage 1.
    """

    if not text:
        return False

    text = str(text).strip()
    if not text:
        return False

    if NON_LATIN_PATTERN.search(text):
        return False

    alphabetic_chars = [char for char in text if char.isalpha()]
    if not alphabetic_chars:
        return True

    ascii_letters = sum(1 for char in alphabetic_chars if char.isascii())
    ascii_ratio = ascii_letters / len(alphabetic_chars)
    return ascii_ratio >= 0.95


def choose_language_value(
    values: Sequence,
    language_priority: Sequence[str],
) -> Optional[str]:
    entries = ensure_list(values)
    normalized_entries: List[Dict] = []

    for entry in entries:
        entry = safe_literal(entry)
        if isinstance(entry, dict):
            normalized_entries.append(entry)

    if not normalized_entries:
        return None

    for language_tag in language_priority:
        for entry in normalized_entries:
            value = str(entry.get("value", "")).strip()
            if entry.get("language_tag") == language_tag and value and is_english_like(value):
                return value

    return None


def choose_plain_values(values: Sequence, language_priority: Sequence[str]) -> List[str]:
    entries = ensure_list(values)
    selected: List[str] = []

    for entry in entries:
        entry = safe_literal(entry)
        if isinstance(entry, dict):
            value = str(entry.get("value", "")).strip()
            language_tag = entry.get("language_tag")
            if value and language_tag in language_priority and is_english_like(value):
                selected.append(value)
        elif isinstance(entry, str) and entry.strip() and is_english_like(entry.strip()):
            selected.append(entry.strip())

    unique_values = list(dict.fromkeys(selected))
    return unique_values


def build_text_description(item: Dict, language_priority: Sequence[str]) -> Optional[str]:
    """
    Build a compact product description from ABO listing metadata.
    We keep it simple and image-relevant for Stage 1.
    """

    item_name = choose_language_value(item.get("item_name"), language_priority)
    brand = choose_language_value(item.get("brand"), language_priority)
    color = choose_language_value(item.get("color"), language_priority)
    style = choose_language_value(item.get("style"), language_priority)
    bullet_points = choose_plain_values(item.get("bullet_point"), language_priority)
    product_types = choose_plain_values(item.get("product_type"), language_priority)

    parts: List[str] = []
    if brand:
        parts.append(brand)
    if item_name:
        parts.append(item_name)
    if color:
        parts.append(f"Color: {color}")
    if style:
        parts.append(f"Style: {style}")
    if product_types:
        parts.append(f"Type: {product_types[0]}")

    image_relevant_bullets = []
    for bullet in bullet_points[:3]:
        lowered = bullet.lower()
        if any(
            keyword in lowered
            for keyword in [
                "color",
                "material",
                "wood",
                "metal",
                "cotton",
                "linen",
                "leather",
                "mesh",
                "size",
                "pattern",
                "design",
                "style",
                "shape",
            ]
        ):
            image_relevant_bullets.append(bullet)

    parts.extend(image_relevant_bullets)

    text = " | ".join(part for part in parts if part and is_english_like(part))
    return text if text else None


def collect_candidate_rows(
    abo_root: Path,
    max_items: int,
    language_priority: Sequence[str],
    prefer_main_image: bool,
    min_text_length: int,
) -> pd.DataFrame:
    listings_dir = abo_root / "listings" / "metadata"
    images_metadata_path = abo_root / "images" / "metadata" / "images.csv.gz"
    images_small_dir = abo_root / "images" / "small"

    if not listings_dir.exists():
        raise FileNotFoundError(f"Listings metadata directory not found: {listings_dir}")
    if not images_small_dir.exists():
        raise FileNotFoundError(f"Small images directory not found: {images_small_dir}")

    image_index = load_image_index(images_metadata_path)
    listing_files = sorted(listings_dir.glob("listings_*.json.gz"))
    if not listing_files:
        raise FileNotFoundError(f"No listing files found in: {listings_dir}")

    candidates: List[Dict] = []

    for listing_file in listing_files:
        for item in tqdm(load_jsonl_gz(listing_file), desc=f"Reading {listing_file.name}"):
            if len(candidates) >= max_items:
                break

            text = build_text_description(item, language_priority)
            if not text or len(text) < min_text_length:
                continue

            item_id = str(item.get("item_id", "")).strip()
            if not item_id:
                continue

            image_ids: List[str] = []
            main_image_id = item.get("main_image_id")
            if isinstance(main_image_id, str) and main_image_id.strip():
                image_ids.append(main_image_id.strip())

            if not prefer_main_image:
                image_ids.extend(
                    image_id.strip()
                    for image_id in ensure_list(item.get("other_image_id"))
                    if isinstance(image_id, str) and image_id.strip()
                )

            image_ids = list(dict.fromkeys(image_ids))
            if not image_ids:
                continue

            added_any = False
            for image_id in image_ids:
                relative_path = image_index.get(image_id)
                if not relative_path:
                    continue

                image_path = images_small_dir / relative_path
                if not image_path.exists():
                    continue

                candidates.append(
                    {
                        "item_id": item_id,
                        "image_id": image_id,
                        "image_path": str(image_path.resolve()),
                        "text": text,
                    }
                )
                added_any = True

                if prefer_main_image:
                    break

            if len(candidates) >= max_items:
                break

        if len(candidates) >= max_items:
            break

    if not candidates:
        raise RuntimeError("No valid ABO image-text candidates were found.")

    dataframe = pd.DataFrame(candidates).drop_duplicates(subset=["item_id", "image_id"]).reset_index(drop=True)
    return dataframe


def sample_negative_texts(
    anchor_item_id: str,
    anchor_text: str,
    text_pool: Sequence[Tuple[str, str]],
    negatives_per_positive: int,
    rng: random.Random,
) -> List[str]:
    candidates = [
        text
        for item_id, text in text_pool
        if item_id != anchor_item_id and text != anchor_text
    ]
    if not candidates:
        return []
    sample_size = min(negatives_per_positive, len(candidates))
    return rng.sample(candidates, k=sample_size)


def build_stage1_pairs(
    candidates_df: pd.DataFrame,
    negatives_per_positive: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = random.Random(random_seed)

    text_pool = list(
        candidates_df[["item_id", "text"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    rows: List[Dict] = []
    for row in candidates_df.itertuples(index=False):
        rows.append(
            {
                "image_path": row.image_path,
                "text": row.text,
                "label": 1,
            }
        )

        negative_texts = sample_negative_texts(
            anchor_item_id=row.item_id,
            anchor_text=row.text,
            text_pool=text_pool,
            negatives_per_positive=negatives_per_positive,
            rng=rng,
        )

        for negative_text in negative_texts:
            rows.append(
                {
                    "image_path": row.image_path,
                    "text": negative_text,
                    "label": 0,
                }
            )

    output_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return output_df


def main() -> None:
    args = parse_args()
    abo_root = Path(args.abo_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    language_priority = [part.strip() for part in args.language_priority.split(",") if part.strip()]

    print("[INFO] Collecting ABO candidates...")
    candidates_df = collect_candidate_rows(
        abo_root=abo_root,
        max_items=args.max_items,
        language_priority=language_priority,
        prefer_main_image=args.prefer_main_image,
        min_text_length=args.min_text_length,
    )

    print(f"[INFO] Collected {len(candidates_df)} valid image-text candidates.")
    print("[INFO] Building positive and negative pairs...")

    output_df = build_stage1_pairs(
        candidates_df=candidates_df,
        negatives_per_positive=args.negatives_per_positive,
        random_seed=args.random_seed,
    )

    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    positive_count = int((output_df["label"] == 1).sum())
    negative_count = int((output_df["label"] == 0).sum())

    print(f"[INFO] Saved Stage 1 CSV to: {output_csv}")
    print(f"[INFO] Total rows      : {len(output_df)}")
    print(f"[INFO] Positive rows   : {positive_count}")
    print(f"[INFO] Negative rows   : {negative_count}")
    print("\n[INFO] Preview:")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
