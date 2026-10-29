"""相机标定模块测试。

自动配对在杂乱真实场景不可靠（方案 2.4 的结论），但给定正确线对时
解析解必须有界且物理合理——这是本模块的测试目标。
"""

from __future__ import annotations

import pytest

from adas.lane.binarize import binarize
from adas.lane.calibrate import calibrate_from_lines, snap_line


def _truth_line_uv(proj, x_m, z0=4.0, z1=30.0):
    """真值地面直线在图像上的两个端点像素。"""
    return proj.project(x_m, z0), proj.project(x_m, z1)


@pytest.mark.integration
def test_calibrate_from_truth_lines(straight_road, cfg, proj):
    """用合成直路的真值边界线标定：pitch、横偏应物理合理。

    configs/default.yaml 的等效参数正是用这套方法在样例数据上标出的，
    本测试同时充当该流程的一致性回归。
    """
    img, _ = straight_road
    mask = binarize(img, cfg.threshold)

    def ab(p0, p1):
        a = (p1[0] - p0[0]) / (p1[1] - p0[1])
        return a, p0[0] - a * p0[1]

    p_l0, p_l1 = _truth_line_uv(proj, -cfg.road.lane_width_m / 2)
    p_r0, p_r1 = _truth_line_uv(proj, +cfg.road.lane_width_m / 2)
    L = calibrate_from_lines(mask, ab(p_l0, p_l1), ab(p_r0, p_r1), camera=cfg.camera)
    assert L is not None, "真值线对标定不应失败"
    assert -10 < L.pitch_deg < 10, f"pitch 越界: {L.pitch_deg}"
    assert abs(L.offset_m) < 1.0, f"直路居中，横偏应接近 0: {L.offset_m}"
    assert abs(L.yaw_deg) < 10


@pytest.mark.integration
def test_snap_line_refines_to_mask(straight_road, cfg, proj):
    """snap_line：人点只要求落在标线附近，吸附后应贴住真值线。"""
    img, _ = straight_road
    mask = binarize(img, cfg.threshold)
    (u0, v0), (u1, v1) = _truth_line_uv(proj, -cfg.road.lane_width_m / 2)
    # 故意偏 10 px 的人点
    snapped = snap_line(mask, (u0 + 10, v0), (u1 + 10, v1))
    assert snapped is not None
    a, b = snapped
    for v in (400, 480, 520):
        u_fit = a * v + b
        # 吸附后的线反投影回地面，横向位置应贴住真值线 x=-1.85 m
        g = proj.unproject(u_fit, v)
        assert g is not None
        assert abs(g[0] - (-cfg.road.lane_width_m / 2)) < 0.3, f"v={v}: 吸附后横偏 {g[0]}"


@pytest.mark.unit
def test_snap_line_rejects_garbage(straight_road, cfg):
    """两点太近（无法定方向）时返回 None。"""
    img, _ = straight_road
    mask = binarize(img, cfg.threshold)
    assert snap_line(mask, (100, 500), (102, 505)) is None
