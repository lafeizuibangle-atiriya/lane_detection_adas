"""二值化冒烟 + v0 基线 + 回归基线（golden file）。

回归基线规则：固定输入的拟合系数存入 tests/golden/，改动算法后用
REGEN_GOLDEN=1 显式重建；测试不会自动更新基线，防止悄悄劣化。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from adas.baseline import BaselineHough
from adas.lane.binarize import binarize

GOLDEN = Path(__file__).parent / "golden" / "fit_coeffs.json"


@pytest.mark.integration
def test_binarize_finds_lane_pixels(straight_road, cfg):
    """合成直路的掩码应有足量车道像素，且噪声占比不高。"""
    img, _ = straight_road
    mask = binarize(img, cfg.threshold)
    n = int(mask.sum() // 255)
    assert n > 3000, f"车道像素过少: {n}"
    ratio = n / mask.size
    assert ratio < 0.10, f"掩码占比过高（噪声嫌疑）: {ratio:.3f}"


@pytest.mark.integration
def test_baseline_finds_lines(straight_road):
    """v0 在合成直路上应找到左右两组线段并给出偏移。"""
    img, _ = straight_road
    r = BaselineHough().process_frame(img)
    assert r["found"], "v0 在合成直路上应检出"
    assert r["offset_m"] is not None
    assert abs(r["offset_m"]) < 1.5, f"v0 偏移离谱: {r['offset_m']}"


def _fit_coeffs(bird, cfg, img):
    from adas.lane.fit import LaneFitter
    mask = binarize(img, cfg.threshold)
    fit = LaneFitter(bird, cfg.lane, cfg.road).fit(bird.warp(mask))
    return {
        "left": None if fit.left is None else [round(float(c), 6) for c in fit.left.coeffs],
        "right": None if fit.right is None else [round(float(c), 6) for c in fit.right.coeffs],
        "offset_m": None if fit.offset_m is None else round(float(fit.offset_m), 4),
        "curvature_m": None if fit.curvature_m is None else round(float(fit.curvature_m), 1),
    }


@pytest.mark.integration
def test_regression_golden(straight_road, curved_road, bird, cfg):
    """拟合系数对照黄金基线。设 REGEN_GOLDEN=1 重建（显式操作，不自动更新）。"""
    actual = {
        "straight": _fit_coeffs(bird, cfg, straight_road[0]),
        "curved": _fit_coeffs(bird, cfg, curved_road[0]),
    }
    if os.environ.get("REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        pytest.skip("黄金基线已重建，重跑以对比")

    assert GOLDEN.exists(), (
        f"缺黄金基线 {GOLDEN}。先跑 REGEN_GOLDEN=1 pytest tests/test_binarize_baseline.py 重建"
    )
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for case in ("straight", "curved"):
        for key in ("offset_m", "curvature_m", "left", "right"):
            g, a = golden[case][key], actual[case][key]
            if g is None or a is None:
                assert g == a, f"{case}.{key} 检出状态改变: golden={g}, now={a}"
                continue
            if isinstance(g, list):
                assert a == pytest.approx(g, abs=2e-3), f"{case}.{key} 系数漂移"
            else:
                # 曲率允许 8% 漂移（拟合本身对像素噪声敏感）；偏移卡 0.1 m
                rel = 0.08 if key == "curvature_m" else 1e-9
                assert a == pytest.approx(g, rel=rel, abs=0.1), f"{case}.{key} 漂移"
