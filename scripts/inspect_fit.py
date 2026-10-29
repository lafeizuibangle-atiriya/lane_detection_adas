"""拟合诊断：把滑窗轨迹、拟合曲线、反投影结果和中间数字放在一张图上。

判断"线找错了"到底是阈值问题还是滑窗问题，看这张图最快：
窗口矩形贴着某条线走 → 阈值没问题，是起点选错或拟合被带偏；
窗口停在一片噪声上 → 阈值太松，先回去调 binarize。

用法：python scripts/inspect_fit.py data/samples/images/solidYellowLeft.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.core.config import load_config  # noqa: E402
from adas.lane.binarize import binarize  # noqa: E402
from adas.lane.fit import LaneFitter  # noqa: E402
from adas.lane.perspective import BirdEyeView  # noqa: E402


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def draw_windows(bird_bgr: np.ndarray, debug_windows: list[dict]) -> np.ndarray:
    out = bird_bgr.copy()
    colors = {"left": (0, 255, 0), "right": (0, 165, 255)}
    for d in debug_windows:
        color = colors.get(d["side"], (200, 200, 200))
        for wd in d["windows"]:
            cv2.rectangle(out, (wd["x_lo"], wd["y_lo"]), (wd["x_hi"], wd["y_hi"]), color, 1)
    return out


def draw_bird_curves(bird_bgr: np.ndarray, fit, bird: BirdEyeView, cfg) -> np.ndarray:
    out = bird_bgr.copy()
    z = np.linspace(cfg.road.z_near_m, cfg.road.z_far_m, 80)
    for poly, color in ((fit.left, (0, 255, 0)), (fit.right, (0, 0, 255))):
        if poly is None:
            continue
        pts = np.stack([poly.x_at(z), z], axis=1)
        px = bird.ground_points_to_bird(pts).astype(np.int32)
        cv2.polylines(out, [px], False, color, 2)
    return out


def draw_overlay(img: np.ndarray, fit, bird: BirdEyeView, cfg) -> np.ndarray:
    out = img.copy()
    proj = bird.projector
    z = np.linspace(cfg.road.z_near_m, cfg.road.z_far_m, 80)

    if fit.left is not None and fit.right is not None:
        left = np.stack([fit.left.x_at(z), z], axis=1)
        right = np.stack([fit.right.x_at(z), z], axis=1)
        poly = np.vstack([left, right[::-1]])
        px = np.array([proj.project(x, zz) for x, zz in poly], dtype=np.int32)
        layer = out.copy()
        cv2.fillPoly(layer, [px], (0, 255, 0))
        out = cv2.addWeighted(layer, 0.30, out, 0.70, 0)

    for poly, color in ((fit.left, (0, 255, 0)), (fit.right, (0, 0, 255))):
        if poly is None:
            continue
        pts = np.stack([poly.x_at(z), z], axis=1)
        px = np.array([proj.project(x, zz) for x, zz in pts], dtype=np.int32)
        cv2.polylines(out, [px], False, color, 3)

    cv2.polylines(out, [bird.roi_polygon], True, (255, 200, 0), 1)
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

    mask = binarize(img, cfg.threshold)
    bm = bird.warp(mask)

    roi_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi_mask, [bird.roi_polygon], 255)
    inside = mask[roi_mask > 0]
    print(f"[{Path(path).name}] ROI 内二值占比 {(inside > 0).mean() * 100:.2f}%  "
          f"| 鸟瞰非零 {(bm > 0).sum()}")

    fitter = LaneFitter(bird, cfg.lane, cfg.road)
    fit = fitter.fit(bm)

    print(f"直方图起点：左={fitter.debug_bases[0]} 右={fitter.debug_bases[1]}"
          f"（鸟瞰宽 {cfg.road.bird_width_px}）")
    for d in fitter.debug_windows:
        counts = [wd["count"] for wd in d["windows"]]
        print(f"  {d['side']:5s} base={d['base']} 逐窗像素数(自下而上) {counts}")

    print(f"检出={fit.found} conf={fit.confidence:.2f} px={fit.n_pixels}")
    print(f"车道宽={fit.width_m} 偏移={fit.offset_m} 曲率={fit.curvature_m}")
    for name, poly in (("左", fit.left), ("右", fit.right)):
        if poly is not None:
            print(f"  {name}线系数(归一化) {np.round(poly.coeffs, 4)} "
                  f"近端 x={float(poly.x_at(cfg.road.z_near_m)):+.2f}m "
                  f"远端 x={float(poly.x_at(cfg.road.z_far_m)):+.2f}m")
            gpts = bird.bird_points_to_ground(
                np.stack([np.full(3, 200.0), np.array([0.0, 450.0, 900.0])], axis=1)
            )
            _ = gpts  # 仅保留接口示例

    cells = [
        draw_windows(to_bgr(bm), fitter.debug_windows),
        draw_bird_curves(to_bgr(bm), fit, bird, cfg),
        draw_overlay(img, fit, bird, cfg),
        cv2.addWeighted(to_bgr(img), 0.55, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.45, 0),
    ]
    labels = ["bird binary + windows", "bird + fitted curves", "overlay", "binary on image"]
    cell_h = max(c.shape[0] for c in cells)
    cell_w = max(c.shape[1] for c in cells)
    tiles = []
    for c, text in zip(cells, labels):
        canvas = np.zeros((cell_h, cell_w, 3), np.uint8)
        canvas[: c.shape[0], : c.shape[1]] = c
        cv2.rectangle(canvas, (0, 0), (cell_w, 24), (0, 0, 0), -1)
        cv2.putText(canvas, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

    out = Path("output/debug_fit.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
