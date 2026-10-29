"""滑窗搜索车道像素，并在米制坐标下拟合二次曲线。

两个容易踩的实现细节，这里都做了处理：

1. **拟合变量归一化**。纵向距离 6~40 米，平方后到 1600，和横向 -4~4 米的量级
   差三个数量级，直接 ``polyfit`` 会得到病态解。这里先把 Z 映射到 [-1, 1] 再拟合，
   曲率计算时用链式法则还原到物理单位。
2. **左右线互相约束**。只有一侧检出时，用车道宽度推另一侧，但标记为外推并降置信度；
   两侧都在时用车道宽度做一致性校验，宽度偏差超过 50% 判为不可信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adas.core.config import LaneConfig, RoadConfig
from adas.lane.perspective import BirdEyeView


@dataclass
class PolyFit:
    """x = f(z) 的二次曲线，系数保存在归一化变量 t=(z-z_mean)/z_scale 下。"""

    coeffs: np.ndarray
    z_mean: float
    z_scale: float
    z_valid: tuple[float, float]
    n_pixels: int = 0
    extrapolated: bool = False

    def _t(self, z):
        return (np.asarray(z, dtype=float) - self.z_mean) / self.z_scale

    def x_at(self, z):
        return np.polyval(self.coeffs, self._t(z))

    def dx_at(self, z):
        d = np.polyval(np.polyder(self.coeffs), self._t(z))
        return d / self.z_scale

    def ddx(self) -> float:
        d2 = np.polyval(np.polyder(self.coeffs, 2), 0.0)
        return float(d2 / (self.z_scale**2))

    def curvature_at(self, z: float) -> float:
        """曲率半径（米）。接近直线时返回 inf。"""
        ddx = self.ddx()
        if abs(ddx) < 1e-9:
            return float("inf")
        dx = float(self.dx_at(z))
        return float((1.0 + dx**2) ** 1.5 / abs(ddx))

    def shifted(self, dx_m: float) -> PolyFit:
        """横向平移后的曲线（常数项加偏移）。"""
        c = self.coeffs.copy()
        c[-1] += dx_m
        return PolyFit(c, self.z_mean, self.z_scale, self.z_valid, self.n_pixels, True)

    def with_smoothing(self, other: PolyFit, alpha: float) -> PolyFit:
        """与上一帧结果做指数平滑，alpha 越大越信当前帧。"""
        if other is None or len(other.coeffs) != len(self.coeffs):
            return self
        c = alpha * self.coeffs + (1.0 - alpha) * other.coeffs
        return PolyFit(c, self.z_mean, self.z_scale, self.z_valid, self.n_pixels, self.extrapolated)


@dataclass
class LaneFit:
    left: PolyFit | None = None
    right: PolyFit | None = None
    offset_m: float | None = None
    curvature_m: float | None = None
    confidence: float = 0.0
    n_pixels: int = 0
    width_m: float | None = None
    stale: bool = False

    @property
    def found(self) -> bool:
        return self.left is not None or self.right is not None

    def sample_ground(self, z_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按给定纵向距离采样左右线的地面坐标，缺失一侧时返回 None。"""
        left_pts = None if self.left is None else np.stack(
            [self.left.x_at(z_values), z_values], axis=1
        )
        right_pts = None if self.right is None else np.stack(
            [self.right.x_at(z_values), z_values], axis=1
        )
        return left_pts, right_pts


class LaneFitter:
    """有状态的拟合器，跨帧保留上一帧结果用于平滑与快速路径。"""

    def __init__(self, bird: BirdEyeView, lane_cfg: LaneConfig, road_cfg: RoadConfig) -> None:
        self.bird = bird
        self.cfg = lane_cfg
        self.road = road_cfg
        self._prev_left: PolyFit | None = None
        self._prev_right: PolyFit | None = None
        self._lost_frames = 0
        # 最近一次搜索的窗口轨迹，供诊断脚本画图用
        self.debug_windows: list[dict] = []
        self.debug_bases: tuple[int | None, int | None] = (None, None)

    def reset(self) -> None:
        self._prev_left = None
        self._prev_right = None
        self._lost_frames = 0

    # ---------- 对外 ----------
    def fit(self, bird_binary: np.ndarray) -> LaneFit:
        left, right = self._search_pair(bird_binary)

        if left is None and right is None:
            self._lost_frames += 1
            if self._lost_frames >= 3:
                self.reset()
                return LaneFit()
            # 短暂丢失：沿用上一帧并标记 stale
            if self._prev_left is not None or self._prev_right is not None:
                return self._compose(self._prev_left, self._prev_right, stale=True)
            return LaneFit()

        self._lost_frames = 0
        left = self._smooth(left, self._prev_left)
        right = self._smooth(right, self._prev_right)
        self._prev_left, self._prev_right = left, right
        return self._compose(left, right, stale=False)

    # ---------- 内部 ----------
    def _smooth(self, cur: PolyFit | None, prev: PolyFit | None) -> PolyFit | None:
        if cur is None or prev is None:
            return cur
        return cur.with_smoothing(prev, self.cfg.ema_alpha)

    def _search_pair(self, binary: np.ndarray):
        """多峰候选 + 车道宽一致性配对。

        每侧取直方图前 k 个峰各自滑窗拟合；枚举左右组合，选多纵深宽度
        最接近车道宽真值的一对。虚线、邻道线会比本车道线更高，单峰直取
        经常锁错；宽度一致性是区分它们的可靠物理量。
        """
        self.debug_windows = []
        peaks_l, peaks_r = self._histogram_peaks(binary)
        cand_l = [self._sliding_window(binary, b, "left") for b in peaks_l]
        cand_r = [self._sliding_window(binary, b, "right") for b in peaks_r]
        cand_l = [c for c in cand_l if c is not None]
        cand_r = [c for c in cand_r if c is not None]

        best = (None, None, np.inf)
        for lf in cand_l or [None]:
            for rf in cand_r or [None]:
                if lf is None and rf is None:
                    continue
                err = self._pair_width_err(lf, rf)
                # 单侧检出时无法验证宽度，给中性罚分，让有像素支持的候选优先
                support = (lf.n_pixels if lf else 0) + (rf.n_pixels if rf else 0)
                total = err + 1e6 / max(support, 1)
                if total < best[2]:
                    best = (lf, rf, total)

        left, right = best[0], best[1]
        # 宽度误差过大且双侧都有候选 → 拒绝，避免把邻道线对当成本车道
        if left is not None and right is not None and self._pair_width_err(left, right) > 1.0:
            return None, None
        return left, right

    def _pair_width_err(self, lf: PolyFit | None, rf: PolyFit | None) -> float:
        """线对在多个纵深上与车道宽真值的平均偏差。单侧返回 0（不参与约束）。"""
        if lf is None or rf is None:
            return 0.0
        errs = []
        for z in self._width_check_zs():
            errs.append(abs((rf.x_at(z) - lf.x_at(z)) - self.road.lane_width_m))
        return float(np.mean(errs))

    def _width_check_zs(self) -> list[float]:
        zs = [self.road.z_near_m + 2.0, self.road.z_near_m + 7.0,
              self.road.z_near_m + 14.0, self.road.z_near_m + 22.0]
        return [z for z in zs if z < self.road.z_far_m]

    def _histogram_peaks(self, binary: np.ndarray, k: int = 3) -> tuple[list[int], list[int]]:
        """在下半幅做列投影，每侧取前 k 个峰（带非极大值抑制）。"""
        h, w = binary.shape
        hist = binary[h // 2:, :].sum(axis=0).astype(np.float64)
        mid = w // 2
        suppress = max(10, w // 30)

        def top_peaks(part: np.ndarray) -> list[int]:
            order: list[int] = []
            h2 = part.copy()
            for _ in range(k):
                if h2.max() <= 0:
                    break
                i = int(np.argmax(h2))
                order.append(i)
                lo, hi = max(0, i - suppress), min(len(h2), i + suppress)
                h2[lo:hi] = 0
            return order

        return [p for p in top_peaks(hist[:mid])], [mid + p for p in top_peaks(hist[mid:])]

    def _sliding_window(self, binary: np.ndarray, base: int | None, side: str) -> PolyFit | None:
        if base is None:
            return None
        h, w = binary.shape
        win_h = max(1, h // self.cfg.n_windows)
        margin = self.cfg.window_margin_px
        nz_y, nz_x = binary.nonzero()

        rows: list[int] = []       # 有足够像素的行号（鸟瞰 y）
        cents: list[float] = []    # 该行的质心 x
        wins: list[dict] = []
        center = base
        for i in range(self.cfg.n_windows):
            y_hi = h - i * win_h
            y_lo = max(0, y_hi - win_h)
            x_lo = max(0, center - margin)
            x_hi = min(w, center + margin)
            sel = (nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= x_lo) & (nz_x < x_hi)
            count = int(sel.sum())
            wins.append(
                {"x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi, "count": count}
            )
            if count >= self.cfg.min_pixels:
                # 逐行质心：远端一行压几十个像素、近端只有几个，
                # 直接对像素拟合会让行密度决定权重，二次项被远端带偏。
                for y in range(y_lo, y_hi):
                    row_sel = sel & (nz_y == y)
                    n_row = int(row_sel.sum())
                    if n_row >= 2:
                        rows.append(y)
                        cents.append(float(nz_x[row_sel].mean()))
                center = int(np.median(nz_x[sel]))
        self.debug_windows.append({"side": side, "base": base, "windows": wins})

        if len(rows) < self.cfg.n_windows * 2:
            return None
        total_px = sum(w["count"] for w in wins)
        if total_px < self.cfg.min_total_pixels:
            return None

        ground = self.bird.bird_points_to_ground(
            np.stack([np.array(cents), np.array(rows, dtype=np.float64)], axis=1)
        )
        return self._polyfit(ground[:, 0], ground[:, 1], total_px)

    def _polyfit(self, x_m: np.ndarray, z_m: np.ndarray, n_pixels: int) -> PolyFit | None:
        z_near, z_far = self.road.z_near_m, self.road.z_far_m
        z_mean = (z_near + z_far) / 2.0
        z_scale = (z_far - z_near) / 2.0
        t = (z_m - z_mean) / z_scale

        order = self.cfg.poly_order
        if np.unique(t).size <= order:
            return None
        coeffs = np.polyfit(t, x_m, order)

        fit = PolyFit(coeffs, z_mean, z_scale, (z_near, z_far), n_pixels)

        # 物理合理性检查：近端横向位置不该跑出 ROI 太多
        x_near = float(fit.x_at(z_near))
        half = self.road.roi_width_m / 2.0
        if abs(x_near) > half + 2.0:
            return None
        if not np.isfinite(fit.curvature_at(z_near)):
            return fit
        if fit.curvature_at(z_near) < 3.0:
            # 半径小于 3 米不是道路，是噪声拟合出的急弯
            return None
        return fit

    def _compose(self, left: PolyFit | None, right: PolyFit | None, stale: bool) -> LaneFit:
        lane_w = self.road.lane_width_m
        extrapolated = False

        if left is not None and right is None:
            right = left.shifted(lane_w)
            extrapolated = True
        elif right is not None and left is None:
            left = right.shifted(-lane_w)
            extrapolated = True

        n_pixels = (left.n_pixels if left else 0) + (right.n_pixels if right else 0)

        z_eval = min(self.road.z_near_m + 4.0, self.road.z_far_m)
        width_m = None
        offset_m = None
        curvature_m = None
        consistency = 0.5

        if left is not None and right is not None:
            xl = float(left.x_at(z_eval))
            xr = float(right.x_at(z_eval))
            width_m = xr - xl
            offset_m = (xl + xr) / 2.0
            err = abs(width_m - lane_w) / lane_w
            consistency = float(np.clip(1.0 - err / 0.5, 0.0, 1.0))
            r_left = left.curvature_at(z_eval)
            r_right = right.curvature_at(z_eval)
            curvature_m = (
                float(np.mean([r_left, r_right]))
                if np.isfinite(r_left) and np.isfinite(r_right)
                else (r_left if np.isfinite(r_left) else r_right)
            )
            if np.isfinite(curvature_m):
                curvature_m = min(curvature_m, self.cfg.max_curvature_m)

        pixel_score = float(np.clip(n_pixels / max(self.cfg.min_total_pixels, 1), 0.0, 1.0))
        confidence = pixel_score * (0.5 + 0.5 * consistency)
        if extrapolated:
            confidence *= 0.6
        if stale:
            confidence *= 0.5

        return LaneFit(
            left=left,
            right=right,
            offset_m=offset_m,
            curvature_m=curvature_m,
            confidence=float(confidence),
            n_pixels=n_pixels,
            width_m=width_m,
            stale=stale,
        )
