# ==============================================================================
# backend.py -- Production-ready SRGAN super-resolution backend module
# ==============================================================================
#
# PURPOSE:
#   Callable backend for SRGAN-based 4x image super-resolution.
#   Designed to be imported by a Streamlit web app or any Python caller.
#   No argparse -- pure callable API.
#
# PIPELINE LOGIC:
#   1. Try to load pretrained SRResNet weights
#   2. If weights load OK  --> run SRGAN inference (deep learning SR)
#   3. If weights missing/corrupt --> fall back to OpenCV bicubic 4x upscale
#      Fallback ALWAYS produces a clean, correct output image.
#
# SRGAN GENERATOR (SRResNet) ROLE:
#   The generator is a deep residual network (SRResNet) that maps a
#   low-resolution input to a photo-realistic 4x high-resolution output.
#   At inference time, only the generator is used (no discriminator needed).
#
# EFFECT OF REDUCING RESIDUAL BLOCKS (num_rcb):
#   The SRResNet trunk is a stack of Residual Convolutional Blocks (RCBs).
#   - Full model : 16 RCBs  (best quality, higher VRAM / latency)
#   - Lightweight:  8 RCBs  (~half the params, faster, slightly softer output)
# ==============================================================================

import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Make sure model.py (SRResNet) is importable from the same project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model import SRResNet  # noqa: E402

# ---------------------------------------------------------------------------
# Default pretrained weights path
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "results", "pretrained_models", "SRGAN_x4-ImageNet.pth.tar"
)

_UPSCALE_FACTOR = 4


# ===========================================================================
# Public API
# ===========================================================================

def run_super_resolution(
    image_path: str,
    output_path: str,
    lightweight: bool = False,
) -> str:
    """Run 4x super-resolution on a single image.

    Tries SRGAN (SRResNet) first. If pretrained weights cannot be loaded,
    automatically falls back to OpenCV bicubic upscaling so the output is
    always a clean, usable image.

    Args:
        image_path:  Path to the input low-resolution image.
        output_path: Where to save the output high-resolution image (.jpg).
        lightweight: True  --> 8 residual blocks  (faster, lighter)
                     False --> 16 residual blocks (full quality, default)

    Returns:
        output_path string (for convenient chaining).
    """

    # ------------------------------------------------------------------
    # 1. Device selection
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backend] Device  : {device}")

    # ------------------------------------------------------------------
    # 2. Model configuration
    # ------------------------------------------------------------------
    num_rcb = 8 if lightweight else 16
    model_label = "lightweight (8 RCBs)" if lightweight else "full (16 RCBs)"
    print(f"[backend] Model   : SRResNet x4 -- {model_label}")

    # ------------------------------------------------------------------
    # 3. Initialise SRResNet generator
    # ------------------------------------------------------------------
    sr_model = SRResNet(
        in_channels=3,
        out_channels=3,
        channels=64,
        num_rcb=num_rcb,
        upscale=_UPSCALE_FACTOR,
    )

    # ------------------------------------------------------------------
    # 4. Try to load pretrained weights
    # ------------------------------------------------------------------
    weights_ok = _load_weights(sr_model, _DEFAULT_WEIGHTS_PATH, device)

    # ------------------------------------------------------------------
    # 5. Route to SRGAN or bicubic fallback
    # ------------------------------------------------------------------
    if weights_ok:
        print("[backend] Method  : Using SRGAN")
        output_path = _run_srgan(sr_model, device, image_path, output_path)
    else:
        print("[backend] Method  : Falling back to Bicubic")
        output_path = bicubic_upscale(image_path, output_path)

    print(f"[backend] Saved   : {os.path.abspath(output_path)}")
    return output_path


# ===========================================================================
# Bicubic fallback
# ===========================================================================

def bicubic_upscale(image_path: str, output_path: str) -> str:
    """Upscale an image 4x using OpenCV bicubic interpolation.

    This is the clean fallback when SRGAN pretrained weights are unavailable.
    Always produces a correct, artifact-free output image.

    Args:
        image_path:  Path to the input image (any format OpenCV supports).
        output_path: Destination path for the saved .jpg result.

    Returns:
        output_path string.
    """
    # Read image (BGR, uint8)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"[backend] Cannot read image: {image_path}")

    h, w = image.shape[:2]
    new_h, new_w = h * _UPSCALE_FACTOR, w * _UPSCALE_FACTOR

    # Bicubic upscale -- cv2.resize handles BGR natively, no channel swap needed
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Ensure output directory exists
    _ensure_dir(output_path)

    # Save as JPEG
    cv2.imwrite(output_path, upscaled)
    return output_path


# ===========================================================================
# Internal helpers
# ===========================================================================

def _run_srgan(
    sr_model: nn.Module,
    device: torch.device,
    image_path: str,
    output_path: str,
) -> str:
    """Run SRResNet inference and save the result."""
    sr_model = sr_model.to(device)
    sr_model.eval()

    # Load and preprocess input image
    input_tensor = _load_image_as_tensor(image_path, device)

    # Inference -- no gradients needed
    with torch.no_grad():
        sr_tensor = sr_model(input_tensor)

    # Clamp to [0, 1] (model does this internally too, but belt-and-suspenders)
    sr_tensor = sr_tensor.clamp(0.0, 1.0)

    # Tensor --> BGR numpy array --> save
    sr_image = _tensor_to_image(sr_tensor)
    _ensure_dir(output_path)
    cv2.imwrite(output_path, sr_image)
    return output_path


def _load_weights(model: nn.Module, weights_path: str, device: torch.device) -> bool:
    """Attempt to load pretrained weights into model.

    Returns True on success, False on any failure (missing file, corrupt
    checkpoint, shape mismatch, etc.). The model is left with random weights
    on failure -- the caller decides what to do next.
    """
    if not os.path.isfile(weights_path) or os.path.getsize(weights_path) < 1024:
        print(f"[backend] Weights file missing or invalid: {weights_path}. Downloading...")
        try:
            import urllib.request
            url = "https://huggingface.co/ChangyuLiu/SRGAN-PyTorch/resolve/main/SRGAN_x4-ImageNet.pth.tar"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            urllib.request.urlretrieve(url, weights_path)
            print("[backend] Download complete.")
        except Exception as e:
            print(f"[backend] WARNING -- Failed to download weights: {e}")
            return False

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,
        )

        # Checkpoint may wrap weights under "state_dict" key
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            print("[backend] WARNING -- Unexpected checkpoint format")
            return False

        # Strip DataParallel ("module.") and torch.compile ("_orig_mod.") prefixes
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[7:]
            if name.startswith("_orig_mod."):
                name = name[10:]
            cleaned[name] = v

        # Only load keys that match by name AND tensor shape
        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in cleaned.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }

        if not filtered:
            print("[backend] WARNING -- No compatible keys found in checkpoint")
            return False

        model_sd.update(filtered)
        model.load_state_dict(model_sd)
        print(f"[backend] Weights : loaded {len(filtered)}/{len(model_sd)} keys [OK]")
        return True

    except Exception as exc:
        print(f"[backend] WARNING -- Failed to load weights: {exc}")
        return False


def _load_image_as_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Read image from disk and return a normalised float32 NCHW tensor (RGB).

    OpenCV reads in BGR; we convert to RGB because SRResNet was trained on
    RGB data. Pixel values are normalised from [0, 255] to [0.0, 1.0].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"[backend] Cannot read image: {image_path}")

    # BGR --> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # [0, 255] uint8 --> [0.0, 1.0] float32
    image = image.astype(np.float32) / 255.0

    # HWC --> CHW --> NCHW (add batch dim)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    return tensor.to(device, non_blocking=True)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert NCHW float tensor [0,1] (RGB) --> HWC uint8 numpy array (BGR).

    The RGB --> BGR conversion is required because cv2.imwrite expects BGR.
    """
    # NCHW --> CHW --> HWC, scale to [0, 255]
    image = (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # RGB --> BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _ensure_dir(file_path: str) -> None:
    """Create parent directories of file_path if they don't exist."""
    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


# ===========================================================================
# Quick smoke-test when run directly (not via argparse)
# ===========================================================================
if __name__ == "__main__":
    test_input  = os.path.join(_PROJECT_ROOT, "figure", "comic.png")
    test_output = os.path.join(_PROJECT_ROOT, "results", "sr_output.jpg")

    if os.path.isfile(test_input):
        result = run_super_resolution(test_input, test_output, lightweight=False)
        print(f"[backend] Done. Output: {result}")
    else:
        print(f"[backend] Test image not found: {test_input}")
        print("[backend] Call manually:")
        print('         run_super_resolution("your_image.png", "output.jpg")')
