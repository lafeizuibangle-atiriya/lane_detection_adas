"""合成道路图像生成器。

按已知的车道几何参数（中心线二次曲线 + 车道宽度）渲染透视图像，
返回图像与真值参数。两个用途：

1. 几何链路验证：拟合结果的宽度、偏移、曲率可以直接和真值比，
   用于区分"几何错了"还是"阈值错了"；
2. pytest 的测试数据工厂：不依赖任何下载数据，CI 里秒级生成。

坐标系与 :mod:`adas.core.camera` 一致：X 向右、Z 向前，车辆沿曲线行驶，
镜头始终对准中心线切线方向时等效于中心线横向偏移，这里简化为
相机不动、道路中心线弯曲（对小曲率，两者对图像的影响近似）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from adas.core.camera import GroundProjector
from adas.core.config import CameraConfig, RoadConfig


@dataclass
class RoadTruth:
    """车道真值：左右线各是一条 x = f(z) 的二次曲线（米制系数）。"""

    left_coeffs: np.ndarray  # [a2, a1, a0]，x = a2*z^2 + a1*z + a0
    right_coeffs: np.ndarray
    curvature_m: float  # 中心线在 z_eval 处的曲率半径
    offset_m: float  # 车辆相对车道中心的横向偏移（z_eval 处）
    z_eval: float


def _curve_coeffs(curvature_m: float, offset_m: float, z_eval: float) -> np.ndarray:
    """构造中心线 x = c(z) = a2*z^2 + a1*z + a0。

    给定 z_eval 处的曲率半径 R 与横向偏移 offset：
      c''(z_eval) = 1/R（符号由 offset 方向决定，这里统一取 1/R）
      c(z_eval) = offset
      c'(z_eval) = 0（车辆行驶方向与道路相切）
    三个条件解出三个系数。
    """
    a2 = 0.5 / curvature_m
    a1 = -a2 * 2.0 * z_eval  # 使 c'(z_eval) = 0
    a0 = offset_m - a2 * z_eval**2 - a1 * z_eval
    return np.array([a2, a1, a0], dtype=np.float64)


def _draw_lane_line(
    img: np.ndarray,
    proj: GroundProjector,
    coeffs: np.ndarray,
    z_start: float,
    z_end: float,
    color_bgr: tuple[int, int, int],
    line_width_m: float,
    n_pts: int = 120,
) -> None:
    """把一条米制曲线画到图像上，线宽按透视随距离衰减。"""
    h, w = img.shape[:2]
    zs = np.linspace(z_start, z_end, n_pts)
    xs = np.polyval(coeffs, zs)
    pts = []
    widths = []
    for x_m, z_m in zip(xs, zs):
        try:
            u, v = proj.project(x_m, z_m)
        except ValueError:
            continue
        if not (0 <= u < w and 0 <= v < h):
            continue
        z_cam = proj._h * proj._sin + z_m * proj._cos
        px_w = max(1, int(round(proj.f * line_width_m / z_cam)))
        pts.append((u, v))
        widths.append(px_w)
    for (u, v), pw in zip(pts, widths):
        cv2.circle(img, (int(round(u)), int(round(v))), pw // 2 + 1, color_bgr, -1)


def render_road(
    camera: CameraConfig | None = None,
    road: RoadConfig | None = None,
    curvature_m: float = 800.0,
    offset_m: float = 0.0,
    noise: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, RoadTruth]:
    """渲染一张含左右车道线的合成道路图。

    Args:
        curvature_m: 中心线曲率半径（米），>1000 近似直道。
        offset_m: 车辆相对车道中心的横向偏移（米），正为偏右。
        noise: 是否叠加路面噪声（模拟沥青纹理）。
    """
    camera = camera or CameraConfig()
    road = road or RoadConfig()
    z_eval = min(road.z_near_m + 4.0, road.z_far_m)

    center = _curve_coeffs(curvature_m, offset_m, z_eval)
    half = road.lane_width_m / 2.0
    left = center.copy()
    left[2] -= half  # x(z) - 1.85
    right = center.copy()
    right[2] += half

    img = np.full((camera.height, camera.width, 3), (70, 60, 50), dtype=np.uint8)
    proj = GroundProjector(camera)

    # 路面：把近/远端两条横向线围成的四边形填成沥青色
    quad = []
    for z_m, c in ((road.z_near_m, center), (road.z_far_m, center)):
        for dx in (-half - road.lane_margin_m, half + road.lane_margin_m):
            try:
                quad.append(proj.project(c[0] + c[1] * z_m + c[2] + dx, z_m))
            except ValueError:
                quad.append((camera.cx, 0 if z_m == road.z_far_m else camera.height))
    cv2.fillPoly(img, [np.array(quad, dtype=np.int32)], (60, 62, 66))

    # 两条车道线：左黄右白（与 Udacity 样例一致）
    z_draw_end = min(road.z_far_m + 20.0, 120.0)
    _draw_lane_line(img, proj, left, road.z_near_m - 2.0, z_draw_end, (0, 210, 220), 0.12)
    _draw_lane_line(img, proj, right, road.z_near_m - 2.0, z_draw_end, (235, 235, 235), 0.12)

    if noise:
        rng = np.random.default_rng(seed)
        grain = rng.integers(-9, 9, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    truth = RoadTruth(
        left_coeffs=left,
        right_coeffs=right,
        curvature_m=float(curvature_m),
        offset_m=float(offset_m),
        z_eval=float(z_eval),
    )
    return img, truth
