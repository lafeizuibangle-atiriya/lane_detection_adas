"""v0 基线：复现原 notebook 的传统方法，作为 v1 的对照。

流程：HSV 颜色阈值 → Canny → 固定比例梯形 ROI → HoughLinesP →
按斜率正负分左右 → 加权一次拟合。

设计意图（方案 4.3）：没有 v0 就没有"改进"的依据。v0 与 v1 跑同一批
视频，对比三个量：中心偏移的帧间抖动（越小越稳）、检出率、单帧耗时。
v0 没有曲率输出——这是方法能力差异，不是实现缺陷，对比表里如实留空。

v0 的中心偏移用"画面底行两线中点相对画面中心的像素偏差"乘以标称
比例换算，几何假设粗糙（这正是 v1 用针孔模型解决的问题）。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 原 notebook 的固定比例 ROI：画面下半部、上底收窄
_ROI_BOTTOM = 1.00
_ROI_TOP = 0.60
_ROI_TOP_HALF_WIDTH = 0.08   # 上底半宽占画面宽度比例
_ROI_BOTTOM_HALF_WIDTH = 0.46


class BaselineHough:
    """v0 基线检测器。无跨帧状态，逐帧独立。"""

    def __init__(self, nominal_lane_width_px: float = 460.0) -> None:
        # 标称：车道在画面底行约占 460 px（960 宽、镜头 78° 的粗略值），
        # 只用于把像素偏差换算成"米"，量级参考，不是标定结果。
        self.nominal_lane_width_px = nominal_lane_width_px

    def process_frame(self, img_bgr: np.ndarray) -> dict:
        h, w = img_bgr.shape[:2]
        t0 = cv2.getTickCount()

        mask = self._color_mask(img_bgr)
        edges = cv2.Canny(mask, 50, 150)
        roi = self._roi_mask(h, w)
        edges = cv2.bitwise_and(edges, roi)

        lines = cv2.HoughLinesP(
            edges, rho=2, theta=np.pi / 180, threshold=50,
            minLineLength=h // 5, maxLineGap=80,
        )
        result = {
            "found": False, "offset_m": None,
            "left": None, "right": None, "n_lines": 0, "elapsed_ms": 0.0,
        }
        if lines is None:
            result["elapsed_ms"] = self._ms(t0)
            return result

        segs = lines[:, 0, :].astype(np.float64)  # (N, 4): x1,y1,x2,y2
        left_segs, right_segs = [], []
        mid_x = w / 2
        for x1, y1, x2, y2 in segs:
            if y2 == y1:
                continue
            slope = (x2 - x1) / (y2 - y1)  # u 对 v 的斜率，与 v1 同约定
            length = float(np.hypot(x2 - x1, y2 - y1))
            # 近似竖直、且起点在对应半幅的段才算
            if abs(slope) < 0.2:
                continue
            xc = (x1 + x2) / 2
            if slope < 0 and xc < mid_x:
                left_segs.append((slope, x1 - slope * y1, length))
            elif slope > 0 and xc > mid_x:
                right_segs.append((slope, x1 - slope * y1, length))

        result["n_lines"] = len(left_segs) + len(right_segs)
        left = self._fit(left_segs, v_ref=h - 1.0)
        right = self._fit(right_segs, v_ref=h - 1.0)
        if left is None and right is None:
            result["elapsed_ms"] = self._ms(t0)
            return result

        result["found"] = True
        result["left"], result["right"] = left, right
        if left is not None and right is not None:
            mid_bottom = ((left[0] * (h - 1) + left[1]) + (right[0] * (h - 1) + right[1])) / 2
            result["offset_m"] = (mid_bottom - mid_x) / self.nominal_lane_width_px * 3.7
        result["elapsed_ms"] = self._ms(t0)
        return result

    @staticmethod
    def _fit(segs: list[tuple[float, float, float]], v_ref: float) -> tuple[float, float] | None:
        """段级加权合并，权重 = 长度。段数太少不可信。

        各段斜率不同，截距不能直接平均；先折算到参考行 v_ref 处的
        横坐标再平均，最后还原成 (a, b)。
        """
        if len(segs) < 2:
            return None
        a = np.array([s[0] for s in segs])
        b = np.array([s[1] for s in segs])
        wgt = np.array([s[2] for s in segs])
        a_bar = float(np.average(a, weights=wgt))
        x_ref = np.average(a * v_ref + b, weights=wgt)
        return a_bar, float(x_ref - a_bar * v_ref)

    @staticmethod
    def _color_mask(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        yellow = cv2.inRange(hsv, (15, 70, 90), (40, 255, 255))
        return cv2.bitwise_or(white, yellow)

    @staticmethod
    def _roi_mask(h: int, w: int) -> np.ndarray:
        pts = np.array([
            (w * (0.5 - _ROI_BOTTOM_HALF_WIDTH), h * _ROI_BOTTOM),
            (w * (0.5 - _ROI_TOP_HALF_WIDTH), h * _ROI_TOP),
            (w * (0.5 + _ROI_TOP_HALF_WIDTH), h * _ROI_TOP),
            (w * (0.5 + _ROI_BOTTOM_HALF_WIDTH), h * _ROI_BOTTOM),
        ], dtype=np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 255)
        return m

    @staticmethod
    def _ms(t0: int) -> float:
        return (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0

    def draw(self, img_bgr: np.ndarray, result: dict) -> np.ndarray:
        """把 v0 的线画出来（红色实线，与 v1 的绿线区分）。"""
        out = img_bgr.copy()
        h = out.shape[0]
        for key in ("left", "right"):
            ab = result.get(key)
            if ab is None:
                continue
            a, b = ab
            p1 = (int(a * (h - 1) + b), h - 1)
            p2 = (int(a * h * 0.6 + b), int(h * 0.6))
            cv2.line(out, p1, p2, (0, 0, 255), 4)
        if result.get("offset_m") is not None:
            cv2.putText(out, f"v0 offset {result['offset_m']:+.2f} m",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out
