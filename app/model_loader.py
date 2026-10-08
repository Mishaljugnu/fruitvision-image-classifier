from functools import lru_cache

import torch
from torchvision import transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, MODEL_PATH


@lru_cache(maxsize=1)
def get_model():
    """Load the trained MiniResNet once and reuse it for later requests."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {MODEL_PATH}. See models/README.md."
        )

    try:
        from models.resnet_manual import MiniResNet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing models/resnet_manual.py. See models/README.md."
        ) from exc

    model = MiniResNet(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
