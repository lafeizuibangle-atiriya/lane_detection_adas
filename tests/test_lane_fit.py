"""透视变换（鸟瞰）与车道拟合的物理量断言。"""

from __future__ import annotations

import numpy as np
import pytest

from adas.lane.binarize import binarize
from adas.lane.fit import LaneFitter


@pytest.mark.unit
def test_bird_roundtrip_px(bird):
    """像素 → 鸟瞰 → 像素，误差 < 1 px（方案 4.1 的自身一致性指标）。"""
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 250], [960, 539], size=(50, 2))
    back = bird.unwarp_points(bird.warp_points(pts))
    assert np.abs(back - pts).max() < 1.0, f"最大往返误差 {np.abs(back - pts).max():.3f} px"


@pytest.mark.unit
def test_bird_scale_is_metric(bird, cfg):
    """鸟瞰图上 1 像素对应的米数应与配置一致（横向、纵向分开验证）。"""
    gx, gz = bird.bird_to_ground(0, 0)
    gx2, gz2 = bird.bird_to_ground(cfg.road.bird_width_px - 1, cfg.road.bird_height_px - 1)
    width_m = abs(gx2 - gx)
    depth_m = abs(gz2 - gz)
    assert width_m == pytest.approx(cfg.road.roi_width_m, rel=0.02)
    assert depth_m == pytest.approx(cfg.road.roi_depth_m, rel=0.02)


def _fit_from_image(bird, cfg, img):
    mask = binarize(img, cfg.threshold)
    return LaneFitter(bird, cfg.lane, cfg.road).fit(bird.warp(mask))


@pytest.mark.integration
def test_straight_road_recovers_geometry(straight_road, bird, cfg):
    """直路：曲率趋于无穷、车道宽接近 3.7 m、偏移接近 0。"""
    img, truth = straight_road
    fit = _fit_from_image(bird, cfg, img)
    assert fit.found, "合成直路必须能检出车道"
    if fit.curvature_m is not None:
        assert fit.curvature_m > cfg.lane.max_curvature_m * 0.9, \
            f"直路曲率不应小于 1800 m，实测 {fit.curvature_m:.0f}"
    assert fit.width_m == pytest.approx(3.7, abs=0.35), f"车道宽 {fit.width_m}"
    assert fit.offset_m == pytest.approx(truth.offset_m, abs=0.3), f"偏移 {fit.offset_m}"


@pytest.mark.integration
def test_curved_road_recovers_geometry(curved_road, bird, cfg):
    """弯道：曲率半径相对误差 < 35%，偏移误差 < 0.4 m。"""
    img, truth = curved_road
    fit = _fit_from_image(bird, cfg, img)
    assert fit.found, "合成弯道必须能检出车道"
    assert fit.curvature_m == pytest.approx(truth.curvature_m, rel=0.35), \
        f"曲率 {fit.curvature_m:.0f} vs 真值 {truth.curvature_m:.0f}"
    assert fit.offset_m == pytest.approx(truth.offset_m, abs=0.4), f"偏移 {fit.offset_m}"


@pytest.mark.integration
@pytest.mark.parametrize("curvature,offset", [
    (800.0, -0.9), (500.0, 0.0), (250.0, 0.5),
])
def test_geometry_parametrized(cfg, bird, curvature, offset):
    """曲率 × 偏移组合扫同一组断言（pytest 参数化）。

    断言分两层：位置（主要物理量，所有工况都卡）在多个纵深上 < 0.25 m；
    曲率只在强弯（R < 350 m，二次项信号远大于噪声）时卡 40%——
    弱弯的 c2 在 36 m 拟合跨度上垂度不足 0.2 m，曲率半径本身
    就不是这套单帧方法的可分辨量，报告里按此口径说明。
    """
    from adas.synthetic.road import render_road
    img, truth = render_road(camera=cfg.camera, curvature_m=curvature,
                             offset_m=offset, seed=3)
    fit = _fit_from_image(bird, cfg, img)
    assert fit.found
    assert fit.offset_m == pytest.approx(truth.offset_m, abs=0.4)
    for z in (8.0, 15.0, 25.0):
        assert fit.left.x_at(z) == pytest.approx(float(np.polyval(truth.left_coeffs, z)),
                                                 abs=0.25), f"左线 @z={z}"
        assert fit.right.x_at(z) == pytest.approx(float(np.polyval(truth.right_coeffs, z)),
                                                  abs=0.25), f"右线 @z={z}"
    if truth.curvature_m < 350.0:
        assert fit.curvature_m == pytest.approx(truth.curvature_m, rel=0.4)
