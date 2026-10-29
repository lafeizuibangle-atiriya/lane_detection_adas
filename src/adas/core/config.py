"""全局配置：相机几何、道路参数、车道线检测参数、阈值。

设计原则：所有几何量用物理单位（米、度）描述，像素坐标由 :mod:`adas.core.camera`
按针孔模型推导。这样换分辨率、换相机安装位置时只改配置，不用重调四点坐标。
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class CameraConfig(BaseModel):
    """相机与安装参数。height_m / pitch_deg 决定地面在画面里的位置。"""

    width: int = 960
    height: int = 540
    # 等效参数：人工指定两条车道边界线后，按"宽度沿 4~30 m 纵深恒定"准则标定，
    # 实测宽度 3.37 m（与 3.7 m 标准差 9%，在高度标称误差内），见 docs/方案设计.md 2.4。
    # θ 与 f 存在互补简并，f 取镜头标称视场角，θ 由宽度恒定准则确定。
    height_m: float = Field(1.10, gt=0.1, description="镜头离地高度（米，标称值）")
    pitch_deg: float = Field(
        -3.8,
        description="等效俯仰角（度），负值表示光轴相对路面略上仰（含路面坡度）",
    )
    hfov_deg: float = Field(78.0, gt=10, lt=150, description="水平视场角（度，镜头标称值）")

    @property
    def focal_px(self) -> float:
        """焦距（像素）。由水平视场角与画面宽度反推。"""
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def horizon_v(self) -> float:
        """地平线在画面中的纵坐标（像素）。越低头的相机地平线越靠上。"""
        return self.cy - self.focal_px * math.tan(math.radians(self.pitch_deg))


class RoadConfig(BaseModel):
    """道路与鸟瞰图参数。z_near/z_far 决定有效检测距离区间。"""

    lane_width_m: float = Field(3.7, gt=1.0, le=6.0, description="单车道宽度（米）")
    lane_margin_m: float = Field(0.40, ge=0.0, description="鸟瞰图在车道两侧额外留的余量")
    z_near_m: float = Field(4.0, gt=0.5, description="有效区间近端（米）")
    z_far_m: float = Field(40.0, gt=5.0, description="有效区间远端（米）")
    bird_width_px: int = Field(400, ge=120, description="鸟瞰图宽度（像素）")
    bird_height_px: int = Field(900, ge=200, description="鸟瞰图高度（像素）")

    @model_validator(mode="after")
    def _check_range(self) -> RoadConfig:
        if self.z_far_m <= self.z_near_m + 2.0:
            raise ValueError("z_far_m 必须比 z_near_m 大至少 2 米")
        return self

    @property
    def roi_width_m(self) -> float:
        """鸟瞰图覆盖的实际横向宽度（米）。"""
        return self.lane_width_m + 2.0 * self.lane_margin_m

    @property
    def roi_depth_m(self) -> float:
        return self.z_far_m - self.z_near_m

    @property
    def meters_per_px_y(self) -> float:
        return self.roi_depth_m / self.bird_height_px

    @property
    def meters_per_px_x(self) -> float:
        return self.roi_width_m / self.bird_width_px


class ThresholdConfig(BaseModel):
    """二值化阈值。

    白色用 HLS 空间（L 亮、S 低），阴影下的白线比 HSV 更稳；
    黄色继续用 HSV，色相区间窄、区分度高。梯度只取竖直方向并限制方向，
    避免把路面横向接缝和阴影边缘当成车道线。
    """

    white_l_min: int = Field(200, ge=0, le=255, description="HLS 亮度下限")
    white_s_max: int = Field(30, ge=0, le=255, description="HLS 饱和度上限")
    yellow_hsv_low: tuple[int, int, int] = (15, 70, 90)
    yellow_hsv_high: tuple[int, int, int] = (40, 255, 255)
    sobel_x_low: int = Field(35, ge=0)
    sobel_x_high: int = Field(255, ge=1, description="梯度幅值上限，一般保持 255 只卡下限")
    grad_dir_low_deg: float = 0.0
    grad_dir_high_deg: float = 35.0
    sobel_ksize: int = Field(3, ge=1, le=7)
    use_gradient: bool = True


class LaneConfig(BaseModel):
    """滑窗搜索与多项式拟合参数。"""

    poly_order: int = Field(2, ge=1, le=3)
    n_windows: int = Field(9, ge=3, le=20)
    window_margin_px: int = Field(70, ge=20)
    min_pixels: int = Field(50, ge=10, description="窗口内少于该像素数就不更新中心")
    min_total_pixels: int = Field(1200, ge=100, description="有效像素总量低于此值判为未检出")
    ema_alpha: float = Field(0.30, gt=0, le=1, description="跨帧系数平滑系数，0 表示不平滑")
    max_curvature_m: float = Field(2000.0, gt=10, description="曲率半径上限，超过按直线处理")
    seek_on_lost: bool = Field(True, description="连续丢失后从直方图重新搜索")


class AppConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    road: RoadConfig = RoadConfig()
    lane: LaneConfig = LaneConfig()
    threshold: ThresholdConfig = ThresholdConfig()


def load_config(path: str | Path) -> AppConfig:
    """从 yaml 读取配置；文件不存在时返回默认值。"""
    import yaml

    p = Path(path)
    if not p.exists():
        return AppConfig()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)
