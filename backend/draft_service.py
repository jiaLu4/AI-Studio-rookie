from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_DIR = PROJECT_ROOT / "storage" / "draft_records"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "storage" / "draft_images"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_field(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def safe_image_extension(filename: Optional[str], content_type: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_IMAGE_EXTENSIONS:
        return suffix

    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


class DraftStore:
    """File-backed draft store for the SmartQC demo.

    Each draft is saved as one JSON record and its image is saved separately.
    This avoids database setup and keeps the demo portable.
    """

    def __init__(
        self,
        record_dir: Path = DEFAULT_RECORD_DIR,
        upload_dir: Path = DEFAULT_UPLOAD_DIR,
    ) -> None:
        self.record_dir = Path(record_dir)
        self.upload_dir = Path(upload_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_draft(
        self,
        *,
        product_name: str = "",
        category: str = "",
        condition: str = "",
        price: str = "",
        short_description: str = "",
        description: str = "",
        decision: str = "",
        image_bytes: Optional[bytes] = None,
        image_filename: Optional[str] = None,
        image_content_type: Optional[str] = None,
        stage1_json: str = "",
        stage2_json: str = "",
        summary_json: str = "",
        local_metrics_json: str = "",
    ) -> Dict[str, Any]:
        draft_id = uuid.uuid4().hex
        now = utc_now_iso()
        stored_image_name: Optional[str] = None
        image_size: Optional[int] = None

        if image_bytes:
            extension = safe_image_extension(image_filename, image_content_type)
            stored_image_name = f"{draft_id}{extension}"
            image_path = self.upload_dir / stored_image_name
            image_path.write_bytes(image_bytes)
            image_size = len(image_bytes)

        draft = {
            "id": draft_id,
            "product_name": product_name,
            "category": category,
            "condition": condition,
            "price": price,
            "short_description": short_description,
            "description": description,
            "decision": decision,
            "image_filename": stored_image_name,
            "image_content_type": image_content_type,
            "image_size": image_size,
            "stage1": parse_json_field(stage1_json),
            "stage2": parse_json_field(stage2_json),
            "summary": parse_json_field(summary_json),
            "local_metrics": parse_json_field(local_metrics_json),
            "created_at": now,
            "updated_at": now,
        }
        self._write_record(draft)
        saved = self.get_draft(draft_id)
        if saved is None:
            raise RuntimeError("Draft was saved but could not be read back.")
        return saved

    def list_drafts(self, limit: int = 30) -> List[Dict[str, Any]]:
        drafts = []
        for path in self.record_dir.glob("*.json"):
            draft = self._read_record(path)
            if draft is not None:
                drafts.append(self._to_summary(draft))

        drafts.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return drafts[:limit]

    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        if not self._is_safe_id(draft_id):
            return None

        draft = self._read_record(self._record_path(draft_id))
        if draft is None:
            return None
        return self._to_detail(draft)

    def image_path_for(self, draft_id: str) -> Optional[Path]:
        draft = self.get_draft(draft_id)
        image_filename = draft.get("image_filename") if draft else None
        if not image_filename:
            return None

        image_path = self.upload_dir / image_filename
        return image_path if image_path.exists() else None

    def health(self) -> Dict[str, Any]:
        return {
            "available": True,
            "storage_type": "json_files",
            "record_dir": str(self.record_dir),
            "upload_dir": str(self.upload_dir),
            "draft_count": len(list(self.record_dir.glob("*.json"))),
        }

    def _record_path(self, draft_id: str) -> Path:
        return self.record_dir / f"{draft_id}.json"

    def _write_record(self, draft: Dict[str, Any]) -> None:
        record_path = self._record_path(draft["id"])
        record_path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_record(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _to_summary(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": draft.get("id"),
            "product_name": draft.get("product_name") or "",
            "category": draft.get("category") or "",
            "condition": draft.get("condition") or "",
            "price": draft.get("price") or "",
            "short_description": draft.get("short_description") or "",
            "decision": draft.get("decision") or "",
            "has_image": bool(draft.get("image_filename")),
            "image_url": self._image_url(draft),
            "created_at": draft.get("created_at"),
            "updated_at": draft.get("updated_at"),
        }

    def _to_detail(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        detail = dict(draft)
        detail["has_image"] = bool(draft.get("image_filename"))
        detail["image_url"] = self._image_url(draft)
        return detail

    def _image_url(self, draft: Dict[str, Any]) -> Optional[str]:
        draft_id = draft.get("id")
        if draft_id and draft.get("image_filename"):
            return f"/drafts/{draft_id}/image"
        return None

    def _is_safe_id(self, draft_id: str) -> bool:
        return bool(draft_id) and all(char in "0123456789abcdef" for char in draft_id.lower())
