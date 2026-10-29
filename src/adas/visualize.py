"""可视化：车道叠加、行人框、告警横幅与 HUD。

全部是纯绘制函数，不改算法状态；颜色语义固定：
    车道区域 半透明绿；外推的车道线 黄色；行人框 青色；
    告警 SAFE 灰 / HINT 蓝 / WARNING 橙 / EMERGENCY 红。
"""

from __future__ import annotations

import cv2
import numpy as np

from adas.core.camera import GroundProjector
from adas.decision.ttc import AlarmLevel, Threat
from adas.lane.fit import LaneFit
from adas.obstacle.pedestrian import Detection

_LEVEL_BGR = {
    AlarmLevel.SAFE: (160, 160, 160),
    AlarmLevel.HINT: (255, 160, 40),
    AlarmLevel.WARNING: (40, 160, 255),
    AlarmLevel.EMERGENCY: (40, 40, 230),
}


def draw_lane(img: np.ndarray, lane: LaneFit, proj: GroundProjector, z_near: float,
              z_far: float, n_samples: int = 24) -> np.ndarray:
    """把车道线与可行驶区域从地面坐标画回原图。缺一侧时按车道宽补齐（lane 已做）。"""
    if not lane.found:
        return img
    zs = np.linspace(z_near, z_far, n_samples)
    left_pts, right_pts = lane.sample_ground(zs)
    if left_pts is None or right_pts is None:
        return img

    poly = np.concatenate([left_pts, right_pts[::-1]], axis=0)
    pix = np.array([proj.project(x, z) for x, z in poly], dtype=np.float32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pix.astype(np.int32)], (60, 180, 60))
    img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)

    color_line = (0, 220, 220) if lane.left and lane.left.extrapolated else (0, 255, 0)
    for pts, c in ((left_pts, color_line), (right_pts, color_line)):
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:], strict=False):
            p0 = tuple(int(round(v)) for v in proj.project(x0, z0))
            p1 = tuple(int(round(v)) for v in proj.project(x1, z1))
            cv2.line(img, p0, p1, c, 3)
    return img


def draw_detections(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """画行人框（青色 + 置信度）。"""
    for det in detections:
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(img, p1, p2, (255, 255, 0), 2)
        cv2.putText(img, f"{det.conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return img


def draw_threats(img: np.ndarray, threats: list[Threat],
                 positions: dict[int, tuple[float, float]]) -> np.ndarray:
    """在轨迹脚点旁标注距离 / TTC / 等级。positions: track_id -> (u, v)。

    多个行人凑在一起时标签会叠，按距离排序后逐个向下错开 18 px。
    """
    for i, th in enumerate(sorted(threats, key=lambda t: t.distance_m)):
        if th.track_id not in positions:
            continue
        u, v = (int(positions[th.track_id][0]), int(positions[th.track_id][1]))
        color = _LEVEL_BGR[th.level]
        cv2.circle(img, (u, v), 5, color, -1)
        ttc_txt = "--" if th.ttc_s is None else f"{th.ttc_s:.1f}s"
        txt = f"#{th.track_id} {th.distance_m:.1f}m TTC {ttc_txt}"
        org_x = min(max(u - 60, 4), img.shape[1] - 240)
        org_y = min(v + 22 + 18 * i, img.shape[0] - 8)
        # 文字后面垫一层半透明底，避免压在行人框上看不清
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (org_x - 2, org_y - th_ - 4), (org_x + tw + 2, org_y + 4),
                      (30, 30, 30), -1)
        cv2.putText(img, txt, (org_x, org_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def draw_banner(img: np.ndarray, level: AlarmLevel) -> int:
    """画面顶部告警条，返回条高（SAFE 不画，返回 0），供 HUD 让位。"""
    if level < AlarmLevel.HINT:
        return 0
    color = _LEVEL_BGR[level]
    h, w = img.shape[:2]
    bar_h = 44 if level >= AlarmLevel.WARNING else 32
    cv2.rectangle(img, (0, 0), (w, bar_h), color, -1)
    text = level.label + (" - BRAKE" if level == AlarmLevel.EMERGENCY else "")
    cv2.putText(img, text, (12, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return bar_h


def draw_hud(img: np.ndarray, lane_offset_m: float | None, curvature_m: float | None,
             lane_conf: float, elapsed_ms: float, y_start: int = 0) -> np.ndarray:
    """左上角状态行：中心偏移、曲率、置信度、耗时。y_start 让出告警横幅的位置。"""
    lines = []
    if lane_offset_m is not None:
        side = "right" if lane_offset_m >= 0 else "left"
        lines.append(f"offset {abs(lane_offset_m):.2f} m {side} of center")
    if curvature_m is not None and np.isfinite(curvature_m):
        lines.append(f"curvature R = {curvature_m:.0f} m")
    lines.append(f"lane conf {lane_conf:.2f}   {elapsed_ms:.0f} ms/frame")
    y = y_start + 26
    for txt in lines:
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y += 22
    return img
