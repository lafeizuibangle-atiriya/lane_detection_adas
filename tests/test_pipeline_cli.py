"""入口路径的集成测试：流水线组装、CLI 落盘、灰度与尺寸归一。

unit 层只测纯函数，这一层补的是"三块拼起来还能跑"以及"命令行输出格式没坏"。
依赖仓库内随附的样例图，不需要联网、不需要视频。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from adas.cli import main as cli_main
from adas.core.config import load_config
from adas.obstacle.pedestrian import Detection, PedestrianDetector
from adas.pipeline import AdasPipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "images" / "solidWhiteRight.jpg"
CONFIG = ROOT / "configs" / "default.yaml"

pytestmark = pytest.mark.integration


class _StubDetector(PedestrianDetector):
    """固定返回一帧里的行人框，避免集成测试依赖 YOLO 权重与 GPU。"""

    name = "stub"

    def __init__(self, boxes: list[Detection]) -> None:
        self._boxes = boxes

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        return list(self._boxes)


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(CONFIG))


@pytest.fixture(scope="module")
def sample_image():
    img = cv2.imread(str(SAMPLE))
    if img is None:
        pytest.skip(f"样例图缺失：{SAMPLE}")
    return img


def test_pipeline_annotates_and_reports_lane(cfg, sample_image):
    pipe = AdasPipeline(cfg, detector=None)
    r = pipe.process_frame(sample_image, t_sec=0.0, frame_idx=0)

    assert r.lane_found is True
    assert r.lane_offset_m is not None
    assert 3.0 < r.lane_width_m < 4.2
    assert r.annotated is not None
    assert r.annotated.shape == sample_image.shape

    # 只用车道分支时，行人相关字段应为空且不报错
    assert r.detections == []
    assert r.threats == []

    js = r.to_json()
    assert js["frame"] == 0
    assert set(js["lane"]) == {"found", "offset_m", "curvature_m", "width_m", "confidence"}


def test_pipeline_records_stage_timings(cfg, sample_image):
    pipe = AdasPipeline(cfg, detector=None)
    pipe.process_frame(sample_image, 0.0, 0)
    timings = pipe.timings_ms
    assert "lane" in timings and "visualize" in timings
    assert all(v >= 0.0 for v in timings.values())
    # 读取后清零，避免跨段累加
    assert pipe.timings_ms == {}


def test_pipeline_accepts_grayscale_input(cfg, sample_image):
    """OpenCV 样例 vtest.avi 是单通道，管线入口必须先转三通道再画。"""
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    pipe = AdasPipeline(cfg, detector=None)
    r = pipe.process_frame(gray, 0.0, 0)
    assert r.annotated is not None
    assert r.annotated.ndim == 3


def test_pipeline_resizes_mismatched_resolution(cfg, sample_image):
    small = cv2.resize(sample_image, (640, 360))
    pipe = AdasPipeline(cfg, detector=None)
    r = pipe.process_frame(small, 0.0, 0)
    assert r.annotated.shape[:2] == (cfg.camera.height, cfg.camera.width)


def test_pipeline_with_detector_draws_threats(cfg, sample_image):
    """给一个虚拟行人框，验证测距→决策→叠加这条链在不崩的前提下产出一致字段。"""
    h, w = sample_image.shape[:2]
    box = Detection(x1=w * 0.45, y1=h * 0.62, x2=w * 0.55, y2=h * 0.92, conf=0.8)
    pipe = AdasPipeline(cfg, detector=_StubDetector([box]))

    out = None
    for i in range(6):  # 多帧才能估出接近速度
        out = pipe.process_frame(sample_image, t_sec=i * 0.1, frame_idx=i)

    js = out.to_json()
    assert js["detections"], "检测框应出现在输出里"
    assert js["detections"][0]["box"] == pytest.approx([box.x1, box.y1, box.x2, box.y2], abs=0.2)
    for th in js["threats"]:
        assert set(th) == {
            "track_id", "distance_m", "lateral_m", "closing_speed_mps", "ttc_s", "level",
        }
        assert th["level"] in {"SAFE", "HINT", "WARNING", "EMERGENCY"}
    assert js["max_level"] in {"SAFE", "HINT", "WARNING", "EMERGENCY"}


def test_cli_run_writes_media_and_json(tmp_path, sample_image):
    code = cli_main([
        "run", str(SAMPLE),
        "-o", str(tmp_path),
        "--config", str(CONFIG),
        "--detector", "none",
    ])
    assert code == 0

    stem = SAMPLE.stem
    assert (tmp_path / f"{stem}_annotated.jpg").exists()
    payload = json.loads((tmp_path / f"{stem}_results.json").read_text(encoding="utf-8"))

    summary = payload["summary"]
    assert summary["frames"] == 1
    assert summary["lane_found_rate"] == 1.0
    assert summary["offset_jitter_m"] is None  # 单帧算不出抖动
    assert "lane" in summary["stage_ms_per_frame"]
    assert len(payload["frames"]) == 1


def test_cli_reports_missing_input(tmp_path):
    code = cli_main([
        "run", str(tmp_path / "not_there.mp4"),
        "-o", str(tmp_path),
        "--config", str(CONFIG),
    ])
    assert code == 1
