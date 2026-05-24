from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CLIP_BACKEND, CLIP_HF_MODEL_ID, CLIP_MODEL_NAME, CLIP_PRETRAINED


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


# Risk-tolerant Stage 1 gatekeeper thresholds.
# Stage 1 should reject obvious invalid/mismatched images, while allowing
# visually valid but lower-quality product photos to continue to Stage 2 MOS.
DEFAULT_POSITIVE_THRESHOLD = 0.22
DEFAULT_NEGATIVE_THRESHOLD = 0.15
DEFAULT_PRODUCT_MARGIN_THRESHOLD = 0.00
DEFAULT_INVALID_SCORE_THRESHOLD = 0.28
DEFAULT_MATCH_WEIGHT = 0.75
DEFAULT_PRODUCT_WEIGHT = 0.20
DEFAULT_INVALID_WEIGHT = 0.50
CLEAR_INVALID_MARGIN = 0.05


@dataclass
class ClipPrediction:
    match_score: float
    decision: str
    reason: str
    raw_match_score: float
    product_score: float
    invalid_score: float
    product_margin: float
    match_margin: float
    final_score: float


def normalize_text(text: str) -> str:
    text = str(text).replace("|", ", ")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def parse_structured_text(text: str) -> Tuple[str, Dict[str, str], List[str]]:
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

    subject = ", ".join(subject_parts).strip(", ") or normalize_text(text)
    return subject, attributes, extras


def build_description_prompts(text: str) -> List[str]:
    normalized_text = normalize_text(text)
    subject, attributes, extras = parse_structured_text(text)

    attribute_fragments: List[str] = []
    if attributes.get("type"):
        attribute_fragments.append(f"type {attributes['type']}")
    if attributes.get("color"):
        attribute_fragments.append(f"color {attributes['color']}")
    if attributes.get("style"):
        attribute_fragments.append(f"style {attributes['style']}")

    attribute_suffix = f", {', '.join(attribute_fragments)}" if attribute_fragments else ""
    extra_fragment = f", details: {extras[0]}" if extras else ""

    candidates = [
        normalized_text,
        f"a product listing image of {subject}",
        f"a clear e-commerce product photo of {subject}{attribute_suffix}",
        f"the main subject is {subject}{attribute_suffix}{extra_fragment}",
    ]

    prompts = list(dict.fromkeys(prompt.strip(" .") + "." for prompt in candidates if prompt.strip()))
    while len(prompts) < 4:
        prompts.append(prompts[-1])
    return prompts[:4]


def normalize_final_score(
    final_score: float,
    negative_threshold: float,
    positive_threshold: float,
) -> float:
    # Map the calibrated decision band onto the UI band:
    # negative threshold -> 60, positive threshold -> 80.
    span = max(positive_threshold - negative_threshold, 1e-6)
    score = 60.0 + ((final_score - negative_threshold) / span) * 20.0
    return float(np.clip(score, 0.0, 100.0))


class ClipConsistencyService:
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_PRETRAINED,
        backend: str = CLIP_BACKEND,
        hf_model_id: str = CLIP_HF_MODEL_ID,
        positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
        negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
        product_margin_threshold: float = DEFAULT_PRODUCT_MARGIN_THRESHOLD,
        invalid_score_threshold: float = DEFAULT_INVALID_SCORE_THRESHOLD,
        match_weight: float = DEFAULT_MATCH_WEIGHT,
        product_weight: float = DEFAULT_PRODUCT_WEIGHT,
        invalid_weight: float = DEFAULT_INVALID_WEIGHT,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.backend = backend
        self.hf_model_id = hf_model_id
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.product_margin_threshold = product_margin_threshold
        self.invalid_score_threshold = invalid_score_threshold
        self.match_weight = match_weight
        self.product_weight = product_weight
        self.invalid_weight = invalid_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None
        self.product_prompt_features = None
        self.invalid_prompt_features = None

    def thresholds(self) -> Dict[str, float]:
        return {
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "product_margin_threshold": self.product_margin_threshold,
            "invalid_score_threshold": self.invalid_score_threshold,
        }

    def load(self) -> None:
        if self.model is not None:
            return

        open_clip_error: Exception | None = None
        if self.backend in {"auto", "open_clip"}:
            try:
                self._load_open_clip()
                return
            except Exception as error:
                if self.backend == "open_clip":
                    raise RuntimeError(
                        "OpenCLIP backend is unavailable. Install open_clip_torch, check model weights, "
                        "or set CLIP_BACKEND=transformers."
                    ) from error
                open_clip_error = error

        if self.backend in {"auto", "transformers"}:
            try:
                self._load_transformers_clip()
                return
            except Exception as error:
                if open_clip_error is not None:
                    raise RuntimeError(
                        f"OpenCLIP backend failed first: {open_clip_error}. "
                        f"Transformers backend also failed: {error}"
                    ) from error
                raise

        raise RuntimeError(f"Unsupported CLIP_BACKEND: {self.backend}")

    def _load_open_clip(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.processor = None
        self.backend = "open_clip"
        self.product_prompt_features = self._encode_text(PRODUCTNESS_PROMPTS)
        self.invalid_prompt_features = self._encode_text(INVALID_IMAGE_PROMPTS)

    def _load_transformers_clip(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(self.hf_model_id).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.hf_model_id)
        model.eval()

        self.model = model
        self.preprocess = None
        self.tokenizer = None
        self.processor = processor
        self.backend = "transformers"
        self.product_prompt_features = self._encode_text(PRODUCTNESS_PROMPTS)
        self.invalid_prompt_features = self._encode_text(INVALID_IMAGE_PROMPTS)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        assert self.model is not None
        if self.backend == "open_clip":
            assert self.tokenizer is not None
            tokenized = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokenized)
        else:
            assert self.processor is not None
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, p=2, dim=-1)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        assert self.model is not None
        if self.backend == "open_clip":
            assert self.preprocess is not None
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_tensor)
        else:
            assert self.processor is not None
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
        return F.normalize(features, p=2, dim=-1)

    def _score_product_validity(self, image_features: torch.Tensor) -> Tuple[float, float, float]:
        assert self.product_prompt_features is not None
        assert self.invalid_prompt_features is not None

        product_scores = image_features @ self.product_prompt_features.T
        invalid_scores = image_features @ self.invalid_prompt_features.T
        product_score = product_scores.max(dim=1).values.item()
        invalid_score = invalid_scores.max(dim=1).values.item()
        product_margin = product_score - invalid_score
        return float(product_score), float(invalid_score), float(product_margin)

    def _combine_scores(self, match_score: float, product_score: float, invalid_score: float) -> float:
        return float(
            self.match_weight * match_score
            + self.product_weight * product_score
            - self.invalid_weight * invalid_score
        )

    def _invalid_evidence_is_clear(self, product_score: float, invalid_score: float) -> bool:
        return (
            invalid_score >= self.invalid_score_threshold + CLEAR_INVALID_MARGIN
            or invalid_score - product_score >= CLEAR_INVALID_MARGIN
        )

    def _message_for_status(self, status: str) -> Tuple[str, str]:
        if status == "consistent":
            return (
                "The image appears to match the product text and looks like a valid product listing image.",
                "Strong image-text alignment with sufficient productness and low invalid-image similarity.",
            )
        if status == "review":
            return (
                "The image-text consistency is borderline. Manual review is recommended.",
                "The final score is between positive and negative thresholds.",
            )
        if status == "inconsistent":
            return (
                "The image does not appear to match the provided product text.",
                "Weak final score or product/text evidence is weaker than invalid-image evidence.",
            )
        if status == "invalid_image":
            return (
                "The uploaded image may not be a valid product listing image.",
                "The image is highly similar to screenshot, poster, collage, selfie, or clutter prompts.",
            )
        return (
            "No product text was provided. Only image validity was checked.",
            "Image-only check cannot determine text-image consistency.",
        )

    def _decide_consistency(
        self,
        match_score: float,
        product_score: float,
        invalid_score: float,
        product_margin: float,
        match_margin: float,
        final_score: float,
    ) -> str:
        if self._invalid_evidence_is_clear(product_score, invalid_score):
            return "invalid_image"
        if (
            final_score >= self.positive_threshold
            and product_margin >= self.product_margin_threshold
            and invalid_score <= self.invalid_score_threshold
        ):
            return "consistent"
        if (
            final_score <= self.negative_threshold
            or (product_margin < 0.0 and match_margin < 0.0)
        ):
            return "inconsistent"
        return "review"

    def _score_image_text(self, image: Image.Image, text: str) -> Tuple[float, float, float, float, float, float]:
        self.load()
        assert self.product_prompt_features is not None
        assert self.invalid_prompt_features is not None

        image_features = self._encode_image(image)
        prompts = build_description_prompts(text)
        prompt_features = self._encode_text(prompts).unsqueeze(0)

        match_score = torch.einsum("bd,bpd->bp", image_features, prompt_features).mean(dim=1).item()
        product_score, invalid_score, product_margin = self._score_product_validity(image_features)
        match_margin = float(match_score - invalid_score)
        final_score = self._combine_scores(float(match_score), product_score, invalid_score)
        return (
            float(match_score),
            product_score,
            invalid_score,
            product_margin,
            match_margin,
            final_score,
        )

    def predict(self, image: Image.Image, text: str) -> ClipPrediction:
        text = normalize_text(text)
        if not text:
            raise ValueError("Product text is empty.")

        (
            match_score,
            product_score,
            invalid_score,
            product_margin,
            match_margin,
            final_score,
        ) = self._score_image_text(image, text)
        decision = self._decide_consistency(
            match_score,
            product_score,
            invalid_score,
            product_margin,
            match_margin,
            final_score,
        )
        _, reason = self._message_for_status(decision)

        return ClipPrediction(
            match_score=normalize_final_score(final_score, self.negative_threshold, self.positive_threshold),
            decision=decision,
            reason=reason,
            raw_match_score=float(match_score),
            product_score=float(product_score),
            invalid_score=float(invalid_score),
            product_margin=float(product_margin),
            match_margin=float(match_margin),
            final_score=float(final_score),
        )

    def check_consistency(self, image: Image.Image, product_name: str = "", description: str = "") -> Dict[str, Any]:
        product_text = " | ".join(part.strip() for part in [product_name, description] if part.strip())
        text = normalize_text(product_text)
        if not text:
            return self.check_image_validity(image)

        (
            match_score,
            product_score,
            invalid_score,
            product_margin,
            match_margin,
            final_score,
        ) = self._score_image_text(image, text)
        status = self._decide_consistency(
            match_score,
            product_score,
            invalid_score,
            product_margin,
            match_margin,
            final_score,
        )
        message, reason = self._message_for_status(status)

        return {
            "ran": True,
            "mode": "text_image",
            "status": status,
            "match_score": match_score,
            "product_score": product_score,
            "invalid_score": invalid_score,
            "product_margin": product_margin,
            "match_margin": match_margin,
            "final_score": final_score,
            "thresholds": self.thresholds(),
            "message": message,
            "reason": reason,
        }

    def check_image_validity(self, image: Image.Image) -> Dict[str, Any]:
        self.load()
        assert self.product_prompt_features is not None
        assert self.invalid_prompt_features is not None

        image_features = self._encode_image(image)
        product_score, invalid_score, product_margin = self._score_product_validity(image_features)

        if self._invalid_evidence_is_clear(product_score, invalid_score):
            status = "invalid_image"
            message, reason = self._message_for_status(status)
        else:
            status = "missing_text_review"
            message, reason = self._message_for_status(status)

        return {
            "ran": True,
            "mode": "image_only",
            "status": status,
            "match_score": None,
            "product_score": product_score,
            "invalid_score": invalid_score,
            "product_margin": product_margin,
            "match_margin": None,
            "final_score": None,
            "thresholds": self.thresholds(),
            "message": message,
            "reason": reason,
        }

    def health(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "backend": self.backend,
            "hf_model_id": self.hf_model_id,
            "device": str(self.device),
            "loaded": str(self.model is not None),
        }
