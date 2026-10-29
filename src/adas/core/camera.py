"""地面投影：把物理坐标（米）与像素坐标互相换算。

采用针孔模型 + 平坦地面假设。世界坐标取相机正下方地面点为原点，
X 向右、Z 向前（沿车辆行驶方向）、Y 向上。相机位于 (0, h, 0)，
光轴水平向前再向下低头 pitch_deg。

这套投影是整个项目的基础：梯形 ROI 的四点、鸟瞰图的米制尺度、
行人脚点的纵向距离，都由这里推导，不靠手工调参。
"""

from __future__ import annotations

import math

import numpy as np

from adas.core.config import CameraConfig


class GroundProjector:
    def __init__(self, camera: CameraConfig) -> None:
        self.cam = camera
        self.f = camera.focal_px
        self.cx = camera.cx
        self.cy = camera.cy
        self._sin = math.sin(math.radians(camera.pitch_deg))
        self._cos = math.cos(math.radians(camera.pitch_deg))
        self._h = camera.height_m

    def project(self, x_m: float, z_m: float) -> tuple[float, float]:
        """地面点 (X, Z) → 像素 (u, v)。"""
        y_cam = self._h * self._cos - z_m * self._sin
        z_cam = self._h * self._sin + z_m * self._cos
        if z_cam <= 1e-6:
            raise ValueError(f"点 (x={x_m}, z={z_m}) 位于相机后方")
        u = self.cx + self.f * x_m / z_cam
        v = self.cy + self.f * y_cam / z_cam
        return u, v

    def unproject(self, u: float, v: float) -> tuple[float, float] | None:
        """像素 (u, v) → 地面点 (X, Z)。落在地平线以上时返回 None。"""
        k = (v - self.cy) / self.f
        denom = k * self._cos + self._sin
        if denom <= 1e-6:
            return None
        z_m = self._h * (self._cos - k * self._sin) / denom
        if z_m <= 0:
            return None
        z_cam = self._h * self._sin + z_m * self._cos
        x_m = (u - self.cx) / self.f * z_cam
        return x_m, z_m

    def distance_of_pixel(self, u: float, v: float) -> float | None:
        """像素点对应的纵向距离（米）。框底中心点用它估距离。"""
        p = self.unproject(u, v)
        return None if p is None else p[1]

    def ground_polygon(self, x_half_m: float, z_near_m: float, z_far_m: float) -> np.ndarray:
        """地面上一个矩形的四个像素角点，顺序为近左、近右、远右、远左。"""
        pts = [
            self.project(-x_half_m, z_near_m),
            self.project(x_half_m, z_near_m),
            self.project(x_half_m, z_far_m),
            self.project(-x_half_m, z_far_m),
        ]
        return np.array(pts, dtype=np.float32)

    def clip_to_image(self, pts: np.ndarray) -> np.ndarray:
        h, w = self.cam.height, self.cam.width
        out = pts.copy()
        out[:, 0] = np.clip(out[:, 0], 0, w - 1)
        out[:, 1] = np.clip(out[:, 1], 0, h - 1)
        return out


def trapezoid_roi(projector: GroundProjector, road_cfg) -> np.ndarray:
    """按道路配置生成梯形 ROI 的四点（像素），用于掩码与透视变换。"""
    half = road_cfg.lane_width_m / 2.0 + road_cfg.lane_margin_m
    return projector.ground_polygon(half, road_cfg.z_near_m, road_cfg.z_far_m)
