"""TTC 决策与测距/跟踪的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from adas.core.camera import GroundProjector
from adas.core.config import AppConfig
from adas.decision.ttc import AlarmLevel, DecisionConfig, TTCAssessor
from adas.obstacle.distance import DistanceEstimator
from adas.obstacle.pedestrian import Detection


def _track(history, dist, lateral=0.0):
    from adas.obstacle.distance import TrackedPedestrian
    return TrackedPedestrian(track_id=1, distance_m=dist, lateral_m=lateral,
                             conf=0.9, bottom_uv=(480, 500), history=history)


@pytest.fixture()
def assessor() -> TTCAssessor:
    return TTCAssessor(DecisionConfig())


@pytest.mark.unit
def test_ttc_constant_speed(assessor):
    """匀速 3 m/s 接近，6 m 处 TTC = 2.0 s → WARNING。"""
    hist = [(t, 24 - 3 * t) for t in np.linspace(0, 6, 61)]
    th = assessor.assess(_track(hist, 6.0), 6.0)
    assert th.closing_speed_mps == pytest.approx(3.0, abs=0.05)
    assert th.ttc_s == pytest.approx(2.0, abs=0.1)
    assert th.level == AlarmLevel.WARNING


@pytest.mark.unit
def test_ttc_emergency(assessor):
    hist = [(t, 8 - 4 * t) for t in np.linspace(0, 1.8, 61)]
    th = assessor.assess(_track(hist, 0.8), 1.8)
    assert th.level == AlarmLevel.EMERGENCY


@pytest.mark.unit
def test_receding_is_safe(assessor):
    """确认在远离：不出 TTC，20 m 处不触发 HINT。"""
    hist = [(t, 10 + 2 * t) for t in np.linspace(0, 5, 51)]
    th = assessor.assess(_track(hist, 20.0), 5.0)
    assert th.ttc_s is None
    assert th.level == AlarmLevel.SAFE


@pytest.mark.unit
def test_stationary_far_is_safe(assessor):
    rng = np.random.default_rng(0)
    hist = [(t, 25 + rng.normal(0, 0.15)) for t in np.linspace(0, 4, 41)]
    th = assessor.assess(_track(hist, 25.0), 4.0)
    assert th.level == AlarmLevel.SAFE


@pytest.mark.unit
def test_distance_floor_emergency(assessor):
    """速度未知但距离 2 m：距离兜底触发 EMERGENCY。"""
    th = assessor.assess(_track([(0, 2.0), (0.1, 2.0)], 2.0), 0.1)
    assert th.level == AlarmLevel.EMERGENCY


@pytest.mark.unit
def test_unknown_speed_near_is_hint(assessor):
    """样本不足速度未知、15 m：宁提示勿漏 → HINT。"""
    th = assessor.assess(_track([(0, 15.0), (0.1, 15.0)], 15.0), 0.1)
    assert th.level == AlarmLevel.HINT


@pytest.mark.unit
def test_speed_needs_min_samples(assessor):
    """少于 3 个样本不给速度，防止两帧差分噪声当真值。"""
    assert assessor.closing_speed([(0, 10), (0.1, 9)]) is None


# ---------- 测距与跟踪 ----------

def _box_at(proj: GroundProjector, x_m: float, z_m: float) -> Detection:
    """在已知地面位置造一个行人框：脚点投影 + 按该距离的人高比例。"""
    u, v = proj.project(x_m, z_m)
    v_head = proj.project(x_m, max(z_m - 0.6, 0.5))[1]
    h_px = max(abs(v - v_head), 8)
    return Detection(u - h_px * 0.15, v - h_px, u + h_px * 0.15, v, 0.9)


@pytest.mark.unit
@pytest.mark.parametrize("z_true", [6.0, 12.0, 25.0])
def test_distance_ground_truth(proj, z_true):
    """脚点反投影测距在合成框上应恢复真值（量化容差随距离放大）。"""
    est = DistanceEstimator(proj, smooth_alpha=1.0)  # 不平滑，看纯几何
    tr = est.update([_box_at(proj, 0.0, z_true)], 0.0)[0]
    tol = 0.05 + 0.02 * z_true
    assert tr.distance_m == pytest.approx(z_true, abs=tol), f"z={z_true}: {tr.distance_m}"


@pytest.mark.unit
def test_tracking_same_pedestrian_single_track(proj):
    """同一行人逐帧接近：只应有一条轨迹，EMA 收敛在真值附近。"""
    est = DistanceEstimator(proj, smooth_alpha=0.5)
    for k, z in enumerate(np.linspace(20, 8, 13)):
        est.update([_box_at(proj, 0.0, z)], k * 0.1)
    assert est.n_tracks == 1
    tr = est.update([], 1.3)[0]
    assert tr.distance_m == pytest.approx(8.0, abs=1.2)  # EMA 滞后 1 帧量级


@pytest.mark.unit
def test_tracking_two_pedestrians(proj):
    """两个横向分隔的行人：两条独立轨迹，距离各自正确。"""
    est = DistanceEstimator(proj)
    est.update([_box_at(proj, -1.5, 10.0), _box_at(proj, 1.5, 20.0)], 0.0)
    est.update([_box_at(proj, -1.5, 9.5), _box_at(proj, 1.5, 19.5)], 0.1)
    assert est.n_tracks == 2
    ds = [t.distance_m for t in est.update([], 0.2)]
    assert min(ds) == pytest.approx(9.0, abs=0.8)
    assert max(ds) == pytest.approx(19.0, abs=1.5)


@pytest.mark.unit
def test_track_expiry(proj):
    """目标消失超过 max_missed 帧后轨迹删除，不留僵尸轨迹。"""
    est = DistanceEstimator(proj, max_missed=3)
    est.update([_box_at(proj, 0.0, 10.0)], 0.0)
    for k in range(5):
        est.update([], 0.1 * (k + 1))
    assert est.n_tracks == 0


@pytest.mark.unit
def test_observation_out_of_range_dropped(proj):
    """反投影不到地面（如误检在天空）的框不进入轨迹。"""
    det = Detection(400, 100, 500, 150, 0.8)  # 画面上部的框
    est = DistanceEstimator(proj)
    tracks = est.update([det], 0.0)
    if tracks:
        assert all(t.distance_m <= 60 for t in tracks)
