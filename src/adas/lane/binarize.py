"""图像二值化：颜色阈值 + 梯度阈值。

白色用 HLS（L 亮、S 低），阴影里的白线比 HSV 阈值稳；
黄色用 HSV，色相区间窄，和路面色区分度高。
梯度只做竖直方向并限制方向角，把路面横向接缝、阴影边缘挡在外面。

两条阈值取并集，再用闭运算把被路面噪声切断的车道线接上。
"""

from __future__ import annotations

import cv2
import numpy as np

from adas.core.config import ThresholdConfig

_KCLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def color_mask(bgr: np.ndarray, thr: ThresholdConfig) -> np.ndarray:
    """白色 + 黄色掩码。"""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(
        hls,
        np.array([0, thr.white_l_min, 0], dtype=np.uint8),
        np.array([180, 255, thr.white_s_max], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(
        hsv,
        np.array(thr.yellow_hsv_low, dtype=np.uint8),
        np.array(thr.yellow_hsv_high, dtype=np.uint8),
    )
    return cv2.bitwise_or(white, yellow)


def gradient_mask(bgr: np.ndarray, thr: ThresholdConfig) -> np.ndarray:
    """竖直梯度幅值 + 方向双重约束。

    车道线在画面里是近垂直边缘：|Gx| 大、|Gy| 小，
    arctan2(|Gy|, |Gx|) 的角度接近 0 度。方向窗默认 [0, 35] 度，
    把路面横向接缝（角度接近 90 度）与阴影斜边挡在外面。
    幅值按峰值归一化，只保留相对最强的边缘。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=thr.sobel_ksize)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=thr.sobel_ksize)

    abs_x = np.abs(sobel_x)
    peak = abs_x.max()
    if peak < 1e-6:
        return np.zeros_like(gray)
    scaled = np.uint8(255.0 * abs_x / peak)
    mag = np.zeros_like(scaled)
    mag[(scaled >= thr.sobel_x_low) & (scaled <= thr.sobel_x_high)] = 255

    with np.errstate(divide="ignore", invalid="ignore"):
        direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x) + 1e-6)
    lo = np.deg2rad(thr.grad_dir_low_deg)
    hi = np.deg2rad(thr.grad_dir_high_deg)
    dir_mask = np.zeros_like(mag)
    dir_mask[(direction >= lo) & (direction <= hi)] = 255

    return cv2.bitwise_and(mag, dir_mask)


def binarize(bgr: np.ndarray, thr: ThresholdConfig) -> np.ndarray:
    """合成二值图（0/255，单通道）。"""
    mask = color_mask(bgr, thr)
    if thr.use_gradient:
        mask = cv2.bitwise_or(mask, gradient_mask(bgr, thr))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KCLOSE)


def binarize_debug(bgr: np.ndarray, thr: ThresholdConfig) -> dict[str, np.ndarray]:
    """返回各分支掩码，用于报告配图与参数排查。"""
    color = color_mask(bgr, thr)
    grad = gradient_mask(bgr, thr) if thr.use_gradient else np.zeros_like(color)
    combined = cv2.bitwise_or(color, grad)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _KCLOSE)
    return {"color": color, "gradient": grad, "combined": combined, "final": closed}
