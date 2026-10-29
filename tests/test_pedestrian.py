"""行人检测接口的无依赖单元测试（不加载 YOLO/torch）。"""

from __future__ import annotations

import pytest

from adas.obstacle.pedestrian import Detection, make_detector


@pytest.mark.unit
def test_detection_bottom_center():
    d = Detection(10, 20, 30, 100, 0.9)
    assert d.bottom_center == (20.0, 100.0)
    assert d.height_px == 80.0


@pytest.mark.unit
def test_make_detector_unknown_raises():
    with pytest.raises(ValueError, match="未知检测后端"):
        make_detector("yolov5")


@pytest.mark.unit
def test_make_detector_hog_builds():
    """HOG 后端不依赖 torch，应能直接构造。"""
    det = make_detector("hog")
    assert det.name == "hog"
