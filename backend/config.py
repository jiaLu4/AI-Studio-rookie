from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOS_MODEL_PATH = Path(
    os.getenv(
        "MOS_MODEL_PATH",
        PROJECT_ROOT / "Models" / "MOS_guess_validation_resnet50_aspect384x512_best.pth",
    )
)

MOS_MODEL_NAME = os.getenv("MOS_MODEL_NAME", "resnet50")
MOS_INPUT_HEIGHT = int(os.getenv("MOS_INPUT_HEIGHT", "384"))
MOS_INPUT_WIDTH = int(os.getenv("MOS_INPUT_WIDTH", "512"))

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
LOCAL_OPENCLIP_WEIGHTS = PROJECT_ROOT / "Models" / "clip-vit-b-32-laion2b" / "open_clip_pytorch_model.bin"
CLIP_PRETRAINED = os.getenv(
    "CLIP_PRETRAINED",
    str(LOCAL_OPENCLIP_WEIGHTS) if LOCAL_OPENCLIP_WEIGHTS.exists() else "laion2b_s34b_b79k",
)
CLIP_BACKEND = os.getenv("CLIP_BACKEND", "auto")
CLIP_HF_MODEL_ID = os.getenv("CLIP_HF_MODEL_ID", "openai/clip-vit-base-patch32")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
