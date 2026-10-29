"""pytest 公共 fixture。

核心原则：unit 层不依赖任何下载数据——道路图由 synthetic.road 程序生成，
真值（曲率、偏移、车道宽）已知，容差断言才有依据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.core.camera import GroundProjector
from adas.core.config import AppConfig
from adas.lane.perspective import BirdEyeView
from adas.synthetic.road import render_road


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture(scope="session")
def proj(cfg) -> GroundProjector:
    return GroundProjector(cfg.camera)


@pytest.fixture(scope="session")
def bird(cfg) -> BirdEyeView:
    return BirdEyeView(cfg.camera, cfg.road)


@pytest.fixture(scope="session")
def straight_road(cfg):
    """近直路（R=5000 m）、居中。返回 (img, truth)。"""
    return render_road(camera=cfg.camera, curvature_m=5000.0, offset_m=0.0, seed=7)


@pytest.fixture(scope="session")
def curved_road(cfg):
    """弯道 R=300 m、向右偏 0.6 m。返回 (img, truth)。"""
    return render_road(camera=cfg.camera, curvature_m=300.0, offset_m=0.6, seed=7)
