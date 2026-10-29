"""完整管线：一帧图 → 车道分支 + 行人分支 + 决策分支 → 标注图与结构化结果。

车道线和行人测距共用同一套针孔投影（GroundProjector）：
车道线拟合在米制地面坐标里做，行人脚点反投影也在这套投影里完成。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from adas.core.camera import GroundProjector
from adas.core.config import AppConfig
from adas.decision.ttc import AlarmLevel, DecisionConfig, TTCAssessor, Threat
from adas.lane.binarize import binarize
from adas.lane.fit import LaneFitter
from adas.lane.perspective import BirdEyeView
from adas.obstacle.distance import DistanceEstimator, TrackedPedestrian
from adas.obstacle.pedestrian import Detection, PedestrianDetector
from adas.visualize import draw_banner, draw_detections, draw_hud, draw_lane, draw_threats


@dataclass
class FrameResult:
    frame_idx: int
    t_sec: float
    lane_offset_m: float | None
    lane_curvature_m: float | None
    lane_width_m: float | None
    lane_confidence: float
    lane_found: bool
    detections: list[Detection] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)
    max_level: AlarmLevel = AlarmLevel.SAFE
    annotated: np.ndarray | None = None

    def to_json(self) -> dict:
        return {
            "frame": self.frame_idx,
            "t_sec": round(self.t_sec, 3),
            "lane": {
                "found": self.lane_found,
                "offset_m": None if self.lane_offset_m is None else round(self.lane_offset_m, 3),
                "curvature_m": None if self.lane_curvature_m is None
                else round(self.lane_curvature_m, 1),
                "width_m": None if self.lane_width_m is None else round(self.lane_width_m, 3),
                "confidence": round(self.lane_confidence, 3),
            },
            "detections": [
                {"conf": round(d.conf, 3),
                 "box": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)]}
                for d in self.detections
            ],
            "threats": [
                {"track_id": th.track_id,
                 "distance_m": round(th.distance_m, 2),
                 "lateral_m": round(th.lateral_m, 2),
                 "closing_speed_mps": round(th.closing_speed_mps, 2),
                 "ttc_s": None if th.ttc_s is None else round(th.ttc_s, 2),
                 "level": th.level.label}
                for th in self.threats
            ],
            "max_level": self.max_level.label,
        }


class AdasPipeline:
    """组合三个分支。跨帧状态（车道平滑、行人轨迹）都在成员里。"""

    def __init__(
        self,
        cfg: AppConfig,
        detector: PedestrianDetector | None = None,
        decision_cfg: DecisionConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.proj = GroundProjector(cfg.camera)
        self.bird = BirdEyeView(cfg.camera, cfg.road)
        self.fitter = LaneFitter(self.bird, cfg.lane, cfg.road)
        self.estimator = DistanceEstimator(self.proj)
        self.assessor = TTCAssessor(decision_cfg)
        self.detector = detector
        self._timings: dict[str, float] = {}

    def reset(self) -> None:
        self.fitter.reset()
        self.estimator.reset()

    @property
    def timings_ms(self) -> dict[str, float]:
        """各阶段平均耗时（自上次读取以来）。读取后清零。"""
        out = {k: v for k, v in self._timings.items()}
        self._timings.clear()
        return out

    def _tick(self, key: str, t0: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0

    def process_frame(self, img_bgr: np.ndarray, t_sec: float, frame_idx: int = 0) -> FrameResult:
        frame_t0 = time.perf_counter()
        if img_bgr.ndim == 2:  # 灰度输入（如 OpenCV 样例 vtest.avi）统一转 BGR
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        if (w, h) != (self.cfg.camera.width, self.cfg.camera.height):
            img_bgr = cv2.resize(img_bgr, (self.cfg.camera.width, self.cfg.camera.height))

        # [A] 车道分支
        t0 = time.perf_counter()
        mask = binarize(img_bgr, self.cfg.threshold)
        bird_binary = self.bird.warp(mask)
        lane = self.fitter.fit(bird_binary)
        self._tick("lane", t0)

        # [B] 行人分支
        detections: list[Detection] = []
        if self.detector is not None:
            t0 = time.perf_counter()
            detections = self.detector.detect(img_bgr)
            self._tick("detect", t0)

        # [C] 测距 + 决策
        t0 = time.perf_counter()
        tracks = self.estimator.update(detections, t_sec)
        threats = self.assessor.assess_all(tracks, t_sec)
        max_level = max((th.level for th in threats), default=AlarmLevel.SAFE)
        self._tick("decision", t0)

        # 可视化
        t0 = time.perf_counter()
        out = img_bgr.copy()
        out = draw_lane(out, lane, self.proj, self.cfg.road.z_near_m, self.cfg.road.z_far_m)
        out = draw_detections(out, detections)
        out = draw_threats(out, threats, {t.track_id: t.bottom_uv for t in tracks})
        bar_h = draw_banner(out, max_level)
        out = draw_hud(out, lane.offset_m, lane.curvature_m, lane.confidence,
                       (time.perf_counter() - frame_t0) * 1000.0, y_start=bar_h)
        self._tick("visualize", t0)

        return FrameResult(
            frame_idx=frame_idx,
            t_sec=t_sec,
            lane_offset_m=lane.offset_m,
            lane_curvature_m=lane.curvature_m,
            lane_width_m=lane.width_m,
            lane_confidence=lane.confidence,
            lane_found=lane.found,
            detections=detections,
            threats=threats,
            max_level=max_level,
            annotated=out,
        )
