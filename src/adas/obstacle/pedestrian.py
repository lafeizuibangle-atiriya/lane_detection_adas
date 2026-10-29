"""行人检测后端。

抽象出统一接口，两个实现：
- YoloDetector：YOLOv8n COCO 预训练权重，只取 person 类。默认后端，精度高，
  CPU 上 960x540 约 100~200 ms/帧。
- HogDetector：OpenCV 自带 HOG+SVM 行人检测器，不依赖 torch，
  用于无 GPU/无权重环境和两个后端的交叉验证。

两者输出统一的 :class:`Detection`，上层测距与决策不关心后端差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """一个行人检测框（像素坐标，原图尺度）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        """框底中心点。测距用它：人站在地面上，脚点是地面与行人的接触点。"""
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1


class PedestrianDetector(ABC):
    """检测后端接口：BGR 图 → 行人框列表。"""

    name: str = "base"

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """返回当前帧的所有行人框。实现必须是纯函数（不保留跨帧状态）。"""


class YoloDetector(PedestrianDetector):
    """YOLOv8n 后端。权重缺失时抛 FileNotFoundError，由 CLI 给出明确提示。"""

    name = "yolo"

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # 延迟导入，HogDetector 用户不需要装 torch

        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        try:
            self._model = YOLO(weights)
        except Exception as e:  # 权重下载失败、文件损坏等
            raise FileNotFoundError(
                f"YOLO 权重不可用（{weights}）：{e}。"
                "首次运行会自动下载 yolov8n.pt，请检查网络后重试，"
                "或改用 --detector hog。"
            ) from e
        self._model.to(device)

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        # classes=[0]：COCO 的 person。verbose=False 关掉逐帧打印。
        results = self._model.predict(
            img_bgr,
            classes=[0],
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(Detection(x1, y1, x2, y2, float(box.conf[0])))
        return out


class HogDetector(PedestrianDetector):
    """HOG+SVM 后端。只适合近景（<20 m），作为对照与兜底。"""

    name = "hog"

    def __init__(self, hit_threshold: float = 0.0, win_stride: tuple[int, int] = (4, 4)) -> None:
        import cv2

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hit = hit_threshold
        self._stride = win_stride

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        import cv2

        # HOG 对大图太慢：等比缩到高 480 再检测，坐标按比例还原
        scale = 1.0
        if img_bgr.shape[0] > 480:
            scale = 480.0 / img_bgr.shape[0]
            small = cv2.resize(
                img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            small = img_bgr
        rects, weights = self._hog.detectMultiScale(
            small,
            hitThreshold=self._hit,
            winStride=self._stride,
            padding=(8, 8),
            scale=1.03,
        )
        out: list[Detection] = []
        for (x, y, w, h), s in zip(rects, weights):
            out.append(
                Detection(x / scale, y / scale, (x + w) / scale, (y + h) / scale, float(s))
            )
        return out


def make_detector(kind: str, **kwargs) -> PedestrianDetector:
    """按名字构造后端，CLI 与配置统一走这里。"""
    backends = {"yolo": YoloDetector, "hog": HogDetector}
    if kind not in backends:
        raise ValueError(f"未知检测后端 {kind!r}，可选：{sorted(backends)}")
    return backends[kind](**kwargs)
