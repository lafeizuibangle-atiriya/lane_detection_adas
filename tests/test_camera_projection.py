"""针孔投影的往返一致性：unit 层的地基，坏了后面全错。"""

from __future__ import annotations

import pytest

from adas.core.config import CameraConfig


@pytest.mark.unit
def test_project_unproject_roundtrip(proj):
    """地面点 → 像素 → 地面点，误差应远小于一个像素对应的距离。"""
    for x, z in [(-2.0, 5.0), (0.0, 10.0), (1.5, 20.0), (-0.7, 35.0), (3.0, 50.0)]:
        u, v = proj.project(x, z)
        x2, z2 = proj.unproject(u, v)
        assert abs(x2 - x) < 1e-6, f"x 往返误差 {x2 - x}"
        assert abs(z2 - z) < 1e-6, f"z 往返误差 {z2 - z}"


@pytest.mark.unit
def test_focal_px_matches_hfov():
    """焦距反推的水平视场角应等于配置值。"""
    cam = CameraConfig(width=960, height=540, hfov_deg=78.0)
    import math
    hfov_back = 2 * math.degrees(math.atan(cam.width / 2 / cam.focal_px))
    assert hfov_back == pytest.approx(78.0, abs=1e-9)


@pytest.mark.unit
def test_horizon_above_camera_bottom(cfg, proj):
    """地平线必须在画面内且车越远 v 越接近地平线（单调）。"""
    assert 0 < cfg.camera.horizon_v < cfg.camera.height
    vs = [proj.project(0.0, z)[1] for z in (5.0, 15.0, 40.0)]
    assert vs[0] > vs[1] > vs[2] > cfg.camera.horizon_v - 1e-6


@pytest.mark.unit
def test_unproject_above_horizon_returns_none(proj):
    """地平线以上的像素没有地面交点，必须返回 None 而不是崩溃。"""
    v_above = proj.cam.horizon_v - 10
    assert proj.unproject(proj.cam.cx, v_above) is None


@pytest.mark.unit
@pytest.mark.parametrize("z", [4.0, 10.0, 25.0, 40.0])
def test_ground_polygon_is_above_horizon(proj, z):
    """远端角点的 v 必须低于（数值小于）近端，且都在画面附近。"""
    poly = proj.ground_polygon(2.0, 2.0, z)
    assert poly[0, 1] > poly[2, 1], "近端行号应大于远端（图像 v 向下增大）"
