"""阈值排查工具：把二值化的每个分支和鸟瞰结果拼成一张图。

调阈值时不要盲改参数，先看这张图，判断噪声来自颜色分支还是梯度分支。
用法：python scripts/inspect_thresholds.py data/samples/images/solidYellowLeft.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.core.config import load_config  # noqa: E402
from adas.lane.binarize import binarize_debug  # noqa: E402
from adas.lane.perspective import BirdEyeView  # noqa: E402


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def pad_to(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """把图左上角贴到固定尺寸画布上，右侧/下方填黑，便于拼接。"""
    out = np.zeros((height, width, 3), np.uint8)
    h = min(height, img.shape[0])
    w = min(width, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def label(img: np.ndarray, text: str, target_h: int | None = None) -> np.ndarray:
    out = to_bgr(img).copy()
    if target_h is not None and out.shape[0] != target_h:
        scale = target_h / out.shape[0]
        out = cv2.resize(out, (max(1, int(out.shape[1] * scale)), target_h))
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/images/solidYellowLeft.jpg"
    cfg = load_config("configs/default.yaml")
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"读不到图片：{path}")

    h, w = img.shape[:2]
    cam = cfg.camera.model_copy(update={"width": w, "height": h})
    bird = BirdEyeView(cam, cfg.road)

    parts = binarize_debug(img, cfg.threshold)
    ratio = {k: float((v > 0).mean() * 100) for k, v in parts.items()}
    for k, v in ratio.items():
        print(f"{k:9s} 非零占比 {v:5.2f}%")

    overlay = img.copy()
    cv2.polylines(overlay, [bird.roi_polygon], True, (0, 255, 0), 2)

    bird_mask = bird.warp(parts["final"])
    bird_color = bird.warp(parts["color"])
    bird_grad = bird.warp(parts["gradient"])

    cell_w, cell_h = w, h
    top = np.hstack([
        pad_to(label(overlay, "original + ROI"), cell_w, cell_h),
        pad_to(label(parts["color"], "color mask"), cell_w, cell_h),
        pad_to(label(parts["gradient"], "gradient mask"), cell_w, cell_h),
    ])
    bot = np.hstack([
        pad_to(label(parts["final"], "combined"), cell_w, cell_h),
        pad_to(label(bird_color, "bird color", h), cell_w, cell_h),
        pad_to(label(bird_mask, "bird combined", h), cell_w, cell_h),
    ])
    grid = np.vstack([top, bot])

    out = Path("output/debug_thresholds.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"已写出 {out}（{grid.shape[1]}x{grid.shape[0]}）")


if __name__ == "__main__":
    main()
