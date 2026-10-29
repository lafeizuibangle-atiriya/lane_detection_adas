"""相机标定：由单帧车道线恢复俯仰角、航向角与横偏。

**可辨识性**（先看这一段，它决定了整个方法的形状）
平坦路面上两条直线车道在图像里是 ``u = a*v + b``，它们约束了相机参数，
但不是所有参数都能从单帧里"解出来"：

============================  ==========================================
量                            能否由单帧直道确定
============================  ==========================================
消失点 (u_vp, v_vp)           能，直接取两线交点，精度约 1~2 px
俯仰角 θ                      能，但必须先知道焦距 f：v_vp = cy - f·tanθ
焦距 f                        不能。它与 θ 强耦合：f 变 40% 时，只要 θ 跟着变，
                              两条线的位置几乎不动。用斜率差反解
                              ``θ = acos(Δa·h/W)`` 在数值上是病态的——
                              Δa 只有 1% 噪声，acos 在 1 附近就放大成十几度
航向偏差 s                    能，u_vp = cx + f·s
横偏 x₀ / 车道宽尺度          能，但只到 W/h 的比值
============================  ==========================================

所以这里把焦距当**输入**（镜头标称水平视场角，购机参数就有），
俯仰角和航向角**从消失点求**——这一步是良态的，实测误差在 0.1° 量级。
W 与 h 只以比值出现，固定标称值；它们整体缩放米制结果，
不影响像素级几何，误差影响见 docs/方案设计.md 4.2。

最后用 Δa 做一次**尺度一致性校验**：按解出的 θ，预测斜率差应为
``W·cosθ/h``；实测值与预测值之比偏离 1 太多，说明标称的 f、W 或 h 有问题，
结果里以 ``scale_ratio`` 报出来，不隐藏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adas.core.config import CameraConfig

NOMINAL_HEIGHT_M = 1.10  # 乘用车前挡风常见安装高度（米）
NOMINAL_LANE_WIDTH_M = 3.70  # 国内车道标准宽度（米）

_PITCH_RANGE = (0.5, 35.0)
_YAW_LIMIT_S = 0.35  # ≈ 19.3°
_OFFSET_LIMIT_M = 3.0


@dataclass
class CalibResult:
    pitch_deg: float
    focal_px: float
    hfov_deg: float
    offset_m: float  # 标定帧车辆相对车道中心的横偏，右正
    yaw_deg: float  # 车辆航向与车道方向的偏差
    residual_px: float  # 掩码点到标定线的平均横向偏差
    scale_ratio: float  # 实测斜率差 / 按标称 W、h 预测的斜率差，理想为 1
    n_pixels: int  # 参与拟合的掩码像素数

    def to_camera_config(self, width: int, height: int) -> CameraConfig:
        return CameraConfig(
            width=width,
            height=height,
            height_m=NOMINAL_HEIGHT_M,
            pitch_deg=round(self.pitch_deg, 3),
            hfov_deg=round(self.hfov_deg, 3),
        )

    @property
    def ok(self) -> bool:
        """尺度一致性在 ±12% 内才认为标称视场角可信。"""
        return abs(self.scale_ratio - 1.0) < 0.12


def _ransac_lines(
    ys: np.ndarray,
    xs: np.ndarray,
    max_lines: int = 8,
    tol_px: float = 3.0,
    strip_px: float = 14.0,
    iters: int = 400,
    min_inliers: int = 150,
    min_dv: float = 60.0,
    seed: int = 0,
) -> list[tuple[float, float, int]]:
    """对掩码像素做顺序 RANSAC，逐条剥离已解释的点，返回 (a, b, 内点数)。

    直接在像素上投票而不是在 Hough 线段上投票：车道线在近处是一条几十像素宽的
    带，Hough 只看到带的边缘，短线段的方向噪声能让斜率偏差 5%；
    像素级投票的内点围绕带的中心对称，斜率误差可压到 1% 以内。

    剥离走廊 strip_px 要大于线带的半宽，否则同一条粗带会被
    反复检出，挤占候选名额（投票容差仍用 3px，只影响内点统计）。
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float, int]] = []
    alive = np.ones(len(ys), dtype=bool)
    for _ in range(max_lines):
        idx = np.flatnonzero(alive)
        if len(idx) < min_inliers:
            break
        ys_k, xs_k = ys[idx], xs[idx]
        best: tuple[int, float, float] | None = None
        for _ in range(iters):
            i, j = int(rng.integers(0, len(ys_k))), int(rng.integers(0, len(ys_k)))
            dv = ys_k[j] - ys_k[i]
            if abs(dv) < min_dv:  # 两点太近，斜率不可信
                continue
            a = (xs_k[j] - xs_k[i]) / dv
            b = xs_k[i] - a * ys_k[i]
            n = int(np.count_nonzero(np.abs(xs_k - (a * ys_k + b)) < tol_px))
            if best is None or n > best[0]:
                best = (n, a, b)
        if best is None or best[0] < min_inliers:
            break
        a0, b0 = best[1], best[2]
        inl = np.abs(xs_k - (a0 * ys_k + b0)) < tol_px
        k = np.polyfit(ys_k[inl].astype(float), xs_k[inl].astype(float), 1)
        a, b = float(k[0]), float(k[1])
        inl = np.abs(xs_k - (a * ys_k + b)) < tol_px
        if int(inl.sum()) < min_inliers:
            break
        out.append((a, b, int(inl.sum())))
        alive[idx[np.abs(xs_k - (a * ys_k + b)) < strip_px]] = False
    return out


def _line_like(mask: np.ndarray, min_px: int = 60, min_rows: int = 20,
               max_dev_px: float = 8.0, keep_ratio: float = 0.85) -> np.ndarray:
    """连通域曲线度过滤：留下"整体贴着一条光滑曲线"的连通域，丢掉团块。

    过曝的树丛、路牌在颜色阈值下也是大片白色，但它们拟合二次曲线的
    横向残差有几十像素；车道线（含单个虚线段）只有几个像素。
    用二次曲线而不是直线，是因为弯道上的实线整体是弯的，
    按直线过滤会把最长的实线误杀，只剩短段。
    """
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = (labels == i).nonzero()
        if ys.max() - ys.min() < min_rows:
            continue
        # 线带宽窄不一，直接对像素拟合会被宽带行带偏；逐行质心才是线的中心
        rows = np.unique(ys)
        cu = np.array([xs[ys == v].mean() for v in rows], dtype=np.float64)
        k = np.polyfit(rows.astype(np.float64), cu, 2)
        if (np.abs(cu - np.polyval(k, rows)) < max_dev_px).mean() >= keep_ratio:
            out[ys, xs] = 255
    return out


def _candidate_lines(mask: np.ndarray, min_support: int = 200) -> list[tuple[float, float, int]]:
    """从清理后的掩码提取候选直线，返回 (a, b, 支持像素数)。

    先用竖直闭运算把被虚线断开的同一条线连成一个连通域，
    每个连通域做逐行质心 + 二次拟合，再取近场一行的切线作为直线候选。
    取切线是因为标定模型假设直道——弯道在近场一小段内近似成立，
    切线的斜率正是近场处两条线的方向，消失点由两条切线相交得到。
    """
    import cv2

    hh, ww = mask.shape
    v_tan = int(hh * 0.92)  # 近场切点：所有候选统一用这一行
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 81))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    out: list[tuple[float, float, int]] = []
    for i in range(1, n):
        ys, xs = (labels == i).nonzero()
        if len(ys) < min_support or ys.max() - ys.min() < 80:
            continue
        rows = np.unique(ys)
        cu = np.array([xs[ys == v].mean() for v in rows], dtype=np.float64)
        k = np.polyfit(rows.astype(np.float64), cu, 2)
        # 近场切线：u = a*v + b
        a = float(2.0 * k[0] * v_tan + k[1])
        b = float(np.polyval(k, v_tan) - a * v_tan)
        out.append((a, b, len(ys)))
    out = [c for c in out if c[2] >= min_support]
    out.sort(key=lambda t: -t[2])
    return out[:8]


def _interior_frac(
    mask: np.ndarray, a_l: float, b_l: float, a_r: float, b_r: float, v_ref: float,
    band: int = 30, margin_px: float = 30.0,
) -> float:
    """两条候选线之间（参考行上方 band 行内）的掩码占比。

    自车道左右边界之间不该再夹着一条标线；若夹着，说明这一对跨过了相邻车道。
    只统计两条线都完整在画面内的行，并让出 margin_px 的带宽，
    否则线带自身的像素会被算成"夹在中间"。
    """
    h, w = mask.shape
    v0 = int(max(0, v_ref - band))
    v1 = int(min(h - 1, v_ref))
    rows, ul_s, ur_s = [], [], []
    for v in range(v0, v1 + 1):
        ul, ur = a_l * v + b_l, a_r * v + b_r
        if margin_px < ul < w - 1 - margin_px and margin_px < ur < w - 1 - margin_px:
            rows.append(v)
            ul_s.append(ul + margin_px)
            ur_s.append(ur - margin_px)
    if len(rows) < 10:
        return 0.0  # 行数不足，无从判断，不否决
    sub = mask[rows]
    total = int(np.count_nonzero(sub))
    if total < 50:
        return 0.0
    ul_a = np.array(ul_s)[:, None]
    ur_a = np.array(ur_s)[:, None]
    xs = np.arange(w)[None, :]
    n_in = int(np.count_nonzero(sub & (xs > ul_a) & (xs < ur_a)))
    return n_in / total


def _ref_row(
    a_l: float, b_l: float, a_r: float, b_r: float, ww: int, hh: int
) -> float | None:
    """两条线都还在画面内的最低一行，作为比较横向间距的参考行。

    用它而不是"标称参数下 6 m 对应的行"，是为了让配对判据完全不依赖标称值：
    车道线在近处会跑出画面，最低的共同可见行总是在有效区间里。
    """
    vs = np.arange(int(hh * 0.5), hh)
    ul = a_l * vs + b_l
    ur = a_r * vs + b_r
    inside = (ul >= 0) & (ul <= ww - 1) & (ur >= 0) & (ur <= ww - 1)
    if not inside.any():
        return None
    return float(vs[inside][-1])


def _residual(ys: np.ndarray, xs: np.ndarray, lines, band_px: float = 30.0) -> float:
    d = np.minimum(
        *[np.abs(xs - (a * ys + b)) for a, b in lines]
    )
    near = d < band_px
    return float(np.mean(d[near])) if near.any() else float("inf")


def calibrate_from_lines(
    mask: np.ndarray,
    line_l: tuple[float, float],
    line_r: tuple[float, float],
    camera: CameraConfig | None = None,
    lane_width_m: float = NOMINAL_LANE_WIDTH_M,
    height_m: float = NOMINAL_HEIGHT_M,
) -> CalibResult | None:
    """由人工指定的左右边界直线标定相机，数学与 :func:`calibrate` 相同。

    自动配对在杂乱场景（虚线、弯道、过曝背景）里经常选错线对，
    而标定本身是一次性操作——人工点两条线、脚本在走廊内吸附掩码质心
    精确拟合，是更可靠的工程做法。参数含义同 :func:`calibrate`。
    """
    hh, ww = mask.shape
    cx, cy = ww / 2.0, hh / 2.0
    cam = camera or CameraConfig(width=ww, height=hh)
    f = cam.focal_px
    ys, xs = mask.nonzero()
    if len(ys) < 400:
        return None

    a_l, b_l = line_l
    a_r, b_r = line_r
    if a_r <= a_l:
        return None
    da = a_r - a_l
    v_vp = (b_l - b_r) / da
    u_vp = a_l * v_vp + b_l
    if not (0.0 < v_vp < cy):
        return None  # 交点得在地平线附近，跑到画面下方说明线选反了
    pitch = math.degrees(math.atan((cy - v_vp) / f))
    if not (_PITCH_RANGE[0] < pitch < _PITCH_RANGE[1]):
        return None
    th = math.radians(pitch)
    s = (u_vp - cx) / f
    if abs(s) > _YAW_LIMIT_S:
        return None
    x_l = (a_l + s * math.sin(th)) * height_m / math.cos(th)
    offset = x_l + lane_width_m / 2.0
    if not (-_OFFSET_LIMIT_M < offset < _OFFSET_LIMIT_M):
        return None
    res = _residual(ys, xs, ((a_l, b_l), (a_r, b_r)))
    if not np.isfinite(res):
        return None
    return CalibResult(
        pitch_deg=pitch,
        focal_px=f,
        hfov_deg=cam.hfov_deg,
        offset_m=offset,
        yaw_deg=math.degrees(math.atan(s)),
        residual_px=res,
        scale_ratio=da / (lane_width_m * math.cos(th) / height_m),
        n_pixels=len(ys),
    )


def snap_line(
    mask: np.ndarray, p_near: tuple[float, float], p_far: tuple[float, float],
    corridor_px: float = 45.0, v_top: float = 0.55,
) -> tuple[float, float] | None:
    """把"人点的大致两个点"精化为掩码里的直线 (a, b)。

    在两点连线的走廊内收集掩码逐行质心（只取下部 v_top 以下的行，
    那里直道假设成立且线最宽），最小二乘拟合；再用拟合结果收紧走廊
    迭代一次。人点只要落在标线附近几十个像素内，结果就由掩码决定，
    与点击精度无关。虚线只有约一半的行有像素，所以行数门槛不能太高。
    """
    h, w = mask.shape
    (u0, v0), (u1, v1) = p_near, p_far
    if abs(v1 - v0) < 30:
        return None
    a0 = (u1 - u0) / (v1 - v0)
    b0 = u0 - a0 * v0
    v_lo = int(max(v_top * h, min(v0, v1) - 20))
    v_hi = int(min(h - 1, max(v0, v1) + 20))

    def fit(a_c: float, b_c: float, corridor: float, min_rows: int):
        rows, cu = [], []
        for v in range(v_lo, v_hi + 1):
            lo, hi = a_c * v + b_c - corridor, a_c * v + b_c + corridor
            i0, i1 = max(0, int(lo)), min(w, int(hi) + 1)
            xs = np.flatnonzero(mask[v, i0:i1]) + i0
            if len(xs) >= 3:
                rows.append(v)
                cu.append(xs.mean())
        if len(rows) < min_rows:
            return None
        k = np.polyfit(np.array(rows, dtype=np.float64), np.array(cu), 1)
        return float(k[0]), float(k[1])

    first = fit(a0, b0, corridor_px, 25)
    if first is None:
        return None
    return fit(first[0], first[1], 15.0, 25)


def calibrate(
    mask: np.ndarray,
    camera: CameraConfig | None = None,
    lane_width_m: float = NOMINAL_LANE_WIDTH_M,
    height_m: float = NOMINAL_HEIGHT_M,
) -> CalibResult | None:
    """从二值车道掩码标定相机。失败（找不到合理线对）返回 None。

    Args:
        mask: 二值图，非零为车道线像素。
        camera: 标称相机配置，其中 hfov_deg 来自镜头规格，作为焦距输入。
        lane_width_m: 车道宽标称值。
        height_m: 镜头离地高度标称值。
    """
    hh, ww = mask.shape
    cx, cy = ww / 2.0, hh / 2.0
    cam = camera or CameraConfig(width=ww, height=hh)
    f = cam.focal_px

    ys, xs = mask.nonzero()
    if len(ys) < 400:
        return None
    # 不按画面中线预分左右：实车装偏时两条自车道边界可能同在一侧。
    # 先做连通域清理（去过曝树丛等团块），再按连通域提候选线，枚举线对。
    clean = _line_like(mask)
    cands = _candidate_lines(clean)
    if len(cands) < 2:
        return None
    ys, xs = clean.nonzero()

    best: tuple[float, CalibResult] | None = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a_l, b_l, n_l = cands[i]
            a_r, b_r, n_r = cands[j]
            # 画面下方更靠左的是左边界
            if a_l * (hh - 1) + b_l > a_r * (hh - 1) + b_r:
                (a_l, b_l), (a_r, b_r) = (a_r, b_r), (a_l, b_l)
            da = a_r - a_l
            v_ref = _ref_row(a_l, b_l, a_r, b_r, ww, hh)
            if v_ref is None:
                continue
            gap = (a_r * v_ref + b_r) - (a_l * v_ref + b_l)
            if da <= 1e-3 or gap < 60.0 or gap > 0.9 * ww:
                continue  # 几乎重合或宽到不像一个车道
            # 自车道应以光轴为中心（横偏不超过约 1.6 m），且两线之间不再夹标线。
            # 这两条判据都不依赖标称尺度，只看图像本身。
            mid_off = abs(((a_l * v_ref + b_l) + (a_r * v_ref + b_r)) * 0.5 - cx) / gap
            if mid_off > 0.45:
                continue
            if _interior_frac(mask, a_l, b_l, a_r, b_r, v_ref) > 0.15:
                continue

            # 消失点 = 两线交点
            v_vp = (b_l - b_r) / da
            u_vp = a_l * v_vp + b_l
            pitch = math.degrees(math.atan((cy - v_vp) / f))
            if not (_PITCH_RANGE[0] < pitch < _PITCH_RANGE[1]):
                continue
            # 消失点共识：同一条路的所有标线（含相邻车道的）都汇聚到同一消失点。
            # 正确线对交出的消失点附近应有其它候选线经过；跨车道配对、
            # 噪声边缘配对则没有。没有第三条线时（画面里只有两条线）不否决。
            if len(cands) >= 3:
                consensus = sum(
                    1
                    for a, b, _ in cands
                    if (a, b) not in ((a_l, b_l), (a_r, b_r))
                    and abs(u_vp - (a * v_vp + b)) / math.hypot(1.0, a) < 30.0
                    and 0.0 < v_vp < cy
                )
                if consensus < 1:
                    continue
            th = math.radians(pitch)
            s = (u_vp - cx) / f
            if abs(s) > _YAW_LIMIT_S:
                continue
            x_l = (a_l + s * math.sin(th)) * height_m / math.cos(th)
            offset = x_l + lane_width_m / 2.0
            if not (-_OFFSET_LIMIT_M < offset < _OFFSET_LIMIT_M):
                continue

            lines = ((a_l, b_l), (a_r, b_r))
            res = _residual(ys, xs, lines)
            if not np.isfinite(res):
                continue
            r = CalibResult(
                pitch_deg=pitch,
                focal_px=f,
                hfov_deg=cam.hfov_deg,
                offset_m=offset,
                yaw_deg=math.degrees(math.atan(s)),
                residual_px=res,
                scale_ratio=da / (lane_width_m * math.cos(th) / height_m),
                n_pixels=n_l + n_r,
            )
            score = res + 40.0 * mid_off
            if best is None or score < best[0]:
                best = (score, r)
    return None if best is None else best[1]
