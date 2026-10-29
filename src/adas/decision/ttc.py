"""TTC（碰撞时间）估计与四级告警。

TTC = 纵向距离 / 接近速度。接近速度由最近若干帧的 (t, z) 样本
做一次最小二乘拟合的斜率取负得到——用拟合而不是相邻两帧差分，
是因为框底一行像素的抖动经几何放大后，单帧差分的速度噪声很大。

四级告警（阈值都可在配置里改）：
    EMERGENCY  TTC ≤ 1.0 s 或距离 < 3 m     立即制动
    WARNING    TTC ≤ 2.5 s 或距离 < 8 m     预警
    HINT       距离 < 20 m 且目标在接近     提示
    SAFE       其余

距离兜底是有意设计的：远距离低接近速度时 TTC 很大，但只要有行人
贴过来就该提示；近距离时即使 TTC 估计不稳（分母小、噪声放大），
距离本身已足以触发高等级告警。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from adas.obstacle.distance import TrackedPedestrian


class AlarmLevel(IntEnum):
    SAFE = 0
    HINT = 1
    WARNING = 2
    EMERGENCY = 3

    @property
    def label(self) -> str:
        return {0: "SAFE", 1: "HINT", 2: "WARNING", 3: "EMERGENCY"}[int(self)]


@dataclass(frozen=True)
class DecisionConfig:
    ttc_emergency_s: float = 1.0
    ttc_warning_s: float = 2.5
    dist_emergency_m: float = 3.0
    dist_warning_m: float = 8.0
    dist_hint_m: float = 20.0
    min_speed_mps: float = 0.3      # 接近速度低于此值视为静止/远离，TTC 记为无穷
    speed_window_s: float = 1.0     # 速度拟合窗口（秒）


@dataclass
class Threat:
    """单个行人的决策结果。"""

    track_id: int
    distance_m: float
    lateral_m: float
    closing_speed_mps: float        # 正值 = 在接近
    ttc_s: float | None             # None = 不在接近或速度太低
    level: AlarmLevel


class TTCAssessor:
    """把轨迹的距离历史换算成 TTC 与告警等级。无内部状态，天然可测。"""

    def __init__(self, cfg: DecisionConfig | None = None) -> None:
        self.cfg = cfg or DecisionConfig()

    def closing_speed(self, history: list[tuple[float, float]]) -> float | None:
        """距离历史的接近速度（m/s，正 = 接近）。样本不足返回 None。

        history: [(t, z), ...]，t 单调递增。
        """
        if len(history) < 3:
            return None
        arr = np.asarray(history, dtype=np.float64)
        t, z = arr[:, 0], arr[:, 1]
        t0 = t[-1] - self.cfg.speed_window_s
        sel = t >= t0
        if sel.sum() < 3:
            sel = np.ones(len(t), dtype=bool)
        tt, zz = t[sel], z[sel]
        span = tt[-1] - tt[0]
        if span < 0.15:  # 窗口太短，速度不可信
            return None
        slope = float(np.polyfit(tt, zz, 1)[0])
        return -slope

    def assess(self, track: TrackedPedestrian, t_sec: float) -> Threat:
        cfg = self.cfg
        speed = self.closing_speed(track.history)
        speed_known = speed is not None
        if not speed_known or speed < cfg.min_speed_mps:
            ttc: float | None = None
            closing = max(speed, 0.0) if speed_known else 0.0
        else:
            closing = speed
            ttc = track.distance_m / speed

        level = self._level(track.distance_m, ttc, speed_known)
        return Threat(
            track_id=track.track_id,
            distance_m=track.distance_m,
            lateral_m=track.lateral_m,
            closing_speed_mps=closing,
            ttc_s=ttc,
            level=level,
        )

    def _level(self, dist: float, ttc: float | None, speed_known: bool) -> AlarmLevel:
        cfg = self.cfg
        if (ttc is not None and ttc <= cfg.ttc_emergency_s) or dist <= cfg.dist_emergency_m:
            return AlarmLevel.EMERGENCY
        if (ttc is not None and ttc <= cfg.ttc_warning_s) or dist <= cfg.dist_warning_m:
            return AlarmLevel.WARNING
        # HINT 要求有接近证据：TTC 可算（确认在接近），或样本太少还看不清（宁提示勿漏）；
        # 已确认不在接近的，只按距离兜底。
        if dist <= cfg.dist_hint_m and (ttc is not None or not speed_known):
            return AlarmLevel.HINT
        return AlarmLevel.SAFE

    def assess_all(self, tracks: list[TrackedPedestrian], t_sec: float) -> list[Threat]:
        return [self.assess(t, t_sec) for t in tracks]
