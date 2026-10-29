"""鸟瞰空间诊断：三路掩码（颜色/梯度/合成）+ 直方图基点。

用法: python scripts/inspect_bird.py [图片路径，默认全部样例]
输出: output/debug_bird_masks.jpg
"""

from __future__ import annotations

import glob
import sys

import cv2
import numpy as np

from adas.core.config import AppConfig, load_config
from adas.lane.binarize import binarize, color_mask, gradient_mask
from adas.lane.fit import LaneFitter
from adas.lane.perspective import BirdEyeView


def roi_mask(shape, bird: BirdEyeView) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [bird.roi_polygon], 255)
    return m


def main() -> None:
    cfg: AppConfig = load_config("configs/default.yaml")
    paths = sys.argv[1:] or sorted(glob.glob("data/samples/images/*.jpg"))

    rows = []
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        cam = cfg.camera.model_copy(update={"width": w, "height": h})
        bird = BirdEyeView(cam, cfg.road)
        keep = roi_mask(img.shape, bird)

        cm = bird.warp(cv2.bitwise_and(color_mask(img, cfg.threshold), keep))
        gm = bird.warp(cv2.bitwise_and(gradient_mask(img, cfg.threshold), keep))
        bm = bird.warp(cv2.bitwise_and(binarize(img, cfg.threshold), keep))

        fitter = LaneFitter(bird, cfg.lane, cfg.road)
        lb, rb = fitter._histogram_bases(bm)
        print(f"{p}: left_base={lb} right_base={rb} 鸟瞰像素={(bm>0).sum()}")

        row = np.hstack([cm, gm, bm])
        mid = row.shape[1] // 2
        cv2.line(row, (mid, 0), (mid, row.shape[0]), (0, 0, 255), 1)
        if lb is not None:
            cv2.circle(row, (lb, row.shape[0] - 20), 7, (0, 255, 255), -1)
        if rb is not None:
            cv2.circle(row, (rb, row.shape[0] - 20), 7, (255, 0, 255), -1)
        name = p.replace("\\", "/").split("/")[-1]
        cv2.putText(row, f"{name}  L={lb} R={rb}", (10, 30), 0, 0.9, (0, 255, 255), 2)
        rows.append(cv2.resize(row, (600, 450)))

    out = np.vstack(rows)
    cv2.imwrite("output/debug_bird_masks.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("saved output/debug_bird_masks.jpg", out.shape)


if __name__ == "__main__":
    main()
