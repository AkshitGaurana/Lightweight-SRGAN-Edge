# ==============================================================================
# metrics.py -- PSNR and SSIM for super-resolution evaluation
# ==============================================================================
#
# Standard SR benchmark methodology:
#   1. Take the user's image as the ground-truth HR image
#   2. Downscale it 4x (bicubic) to simulate a real LR capture
#   3. Upscale the LR back to HR using each method
#   4. Compute PSNR / SSIM between the SR result and the original HR
#
# This is identical to how Set5 / Set14 / BSD100 benchmarks are evaluated.
# ==============================================================================

import time
import cv2
import numpy as np


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two uint8 BGR images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index between two uint8 BGR images.
    Uses the standard constants C1, C2 from Wang et al. (2004).
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    # Per-channel SSIM, then average
    ssim_vals = []
    for ch in range(img1.shape[2]):
        c1 = img1[:, :, ch]
        c2 = img2[:, :, ch]

        mu1 = cv2.filter2D(c1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(c2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(c1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(c2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12   = cv2.filter2D(c1 * c2, -1, window)[5:-5, 5:-5] - mu1_mu2

        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        ssim_vals.append(float(np.mean(ssim_map)))

    return float(np.mean(ssim_vals))


def evaluate_method(
    hr_image: np.ndarray,
    sr_func,
    label: str,
) -> dict:
    """Run sr_func on a synthetically downscaled LR image and measure quality.

    Args:
        hr_image: Ground-truth HR image (BGR uint8).
        sr_func:  Callable(lr_image) -> sr_image (BGR uint8).
        label:    Method name for logging.

    Returns:
        Dict with keys: method, psnr, ssim, time_ms
    """
    h, w = hr_image.shape[:2]
    lh, lw = h // 4, w // 4

    # Simulate real-world LR acquisition: bicubic downsample
    lr_image = cv2.resize(hr_image, (lw, lh), interpolation=cv2.INTER_CUBIC)

    # Run the SR method and time it
    t0 = time.perf_counter()
    sr_image = sr_func(lr_image)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Crop to same size (model output might differ by 1-2 px due to padding)
    sr_h, sr_w = sr_image.shape[:2]
    crop_h, crop_w = min(h, sr_h), min(w, sr_w)
    hr_crop = hr_image[:crop_h, :crop_w]
    sr_crop = sr_image[:crop_h, :crop_w]

    psnr = compute_psnr(hr_crop, sr_crop)
    ssim = compute_ssim(hr_crop, sr_crop)

    return {
        "method":   label,
        "psnr":     round(psnr, 2),
        "ssim":     round(ssim, 4),
        "time_ms":  round(elapsed_ms, 1),
    }
