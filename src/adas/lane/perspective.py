"""透视变换：原图 ↔ 米制鸟瞰图。

梯形的四点由 :class:`GroundProjector` 按物理参数推出（近端 6 米、远端 40 米、
左右各留 0.4 米余量），因此换分辨率或换相机安装高度只需要改配置。

鸟瞰图的 x/y 尺度不同（横幅 8.2 米铺 400 像素、纵深 34 米铺 900 像素），
这是透视变换的固有性质。所以**拟合与曲率计算一律在米制坐标下做**，
鸟瞰图像素只用于滑窗搜索和可视化。
"""

from __future__ import annotations

import cv2
import numpy as np

from adas.core.camera import GroundProjector, trapezoid_roi
from adas.core.config import CameraConfig, RoadConfig


class BirdEyeView:
    def __init__(self, camera: CameraConfig, road: RoadConfig) -> None:
        self.cam = camera
        self.road = road
        self.projector = GroundProjector(camera)

        self.src = trapezoid_roi(self.projector, road)
        w, h = road.bird_width_px, road.bird_height_px
        # 顺序与 ground_polygon 一致：近左、近右、远右、远左
        self.dst = np.float32([[0, h], [w, h], [w, 0], [0, 0]])

        self._check_bounds()
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def _check_bounds(self) -> None:
        h, w = self.cam.height, self.cam.width
        if self.src[:, 1].max() > h or self.src[:, 1].min() < 0:
            raise ValueError(
                f"梯形 ROI 超出画面高度（v={self.src[:, 1]}，画面高 {h}）。"
                "请调小 road.z_far_m 或调大 camera.pitch_deg"
            )
        if self.src[:, 0].min() < 0 or self.src[:, 0].max() > w:
            raise ValueError(
                f"梯形 ROI 超出画面宽度（u={self.src[:, 0]}，画面宽 {w}）。"
                "请调小 road.lane_margin_m 或调大 camera.hfov_deg"
            )

    # ---------- 图像 ----------
    def warp(self, img: np.ndarray) -> np.ndarray:
        w, h = self.road.bird_width_px, self.road.bird_height_px
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def unwarp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.Minv, (self.cam.width, self.cam.height))

    # ---------- 点 ----------
    def warp_points(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.M).reshape(-1, 2)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Minv).reshape(-1, 2)

    # ---------- 米制换算 ----------
    @property
    def _half_width(self) -> float:
        return self.road.roi_width_m / 2.0

    def bird_to_ground(self, x_b: float, y_b: float) -> tuple[float, float]:
        """鸟瞰图像素 → 地面 (X 横向米, Z 纵向米)。"""
        x_m = -self._half_width + x_b * self.road.meters_per_px_x
        z_m = self.road.z_far_m - y_b * self.road.meters_per_px_y
        return x_m, z_m

    def ground_to_bird(self, x_m: float, z_m: float) -> tuple[float, float]:
        x_b = (x_m + self._half_width) / self.road.meters_per_px_x
        y_b = (self.road.z_far_m - z_m) / self.road.meters_per_px_y
        return x_b, y_b

    def bird_points_to_ground(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x_m = -self._half_width + pts[:, 0] * self.road.meters_per_px_x
        z_m = self.road.z_far_m - pts[:, 1] * self.road.meters_per_px_y
        return np.stack([x_m, z_m], axis=1)

    def ground_points_to_bird(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x_b = (pts[:, 0] + self._half_width) / self.road.meters_per_px_x
        y_b = (self.road.z_far_m - pts[:, 1]) / self.road.meters_per_px_y
        return np.stack([x_b, y_b], axis=1)

    # ---------- 单点距离 ----------
    def pixel_distance(self, u: float, v: float) -> float | None:
        """像素点落到地面的纵向距离（米）。地平线以上返回 None。"""
        return self.projector.distance_of_pixel(u, v)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """像素点 → 地面坐标。先投影到鸟瞰图再换算，与 warp 结果保持一致。"""
        x_b, y_b = self.warp_points([[u, v]])[0]
        if not (0 <= y_b <= self.road.bird_height_px):
            return None
        return self.bird_to_ground(x_b, y_b)

    @property
    def roi_polygon(self) -> np.ndarray:
        """梯形 ROI 的整数像素坐标，用于画掩码。"""
        return self.src.astype(np.int32)
