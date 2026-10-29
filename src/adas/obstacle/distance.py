"""单目测距与跨帧跟踪。

测距原理：行人站在平坦地面上，检测框的底边中心点近似是"脚点"，
脚点经针孔模型反投影（:meth:`GroundProjector.unproject`）直接得到
纵向距离 Z 与横向位置 X。不需要知道行人的真实身高——这是选脚点
而不是用框高估计距离的原因。

跨帧部分解决两个问题：
1. 像素量化：框底一行的抖动在远距离被几何放大（v 差 1 px，30 m 处
   距离差 1 m 以上），所以每条轨迹对距离做 EMA 平滑。
2. 轨迹关联：逐帧最近邻匹配（按脚点横向位置），没有用 IoU——
   相邻帧行人框位移小而框尺寸稳定，脚点匹配在单人场景足够；
   多人交叉换位时偶发 ID 跳变，如实写入局限。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from adas.core.camera import GroundProjector
from adas.obstacle.pedestrian import Detection

_ids = itertools.count(1)


@dataclass
class TrackedPedestrian:
    """一条行人轨迹的当帧状态。"""

    track_id: int
    distance_m: float                      # 平滑后的纵向距离
    lateral_m: float                       # 横向位置（正 = 行进方向右侧）
    conf: float
    bottom_uv: tuple[float, float]         # 脚点像素坐标
    age: int = 1                           # 已连续出现的帧数
    history: list[tuple[float, float]] = field(default_factory=list)  # (t, z) 样本


@dataclass
class _TrackState:
    """内部轨迹状态（不对外）。missed 连续未匹配的帧数，超限删除。"""

    track_id: int
    last_u: float
    last_v: float
    z_smooth: float
    x_smooth: float
    conf: float
    age: int
    history: list[tuple[float, float]]
    last_t: float = 0.0
    missed: int = 0


class DistanceEstimator:
    """检测框 → 平滑距离轨迹。

    参数：
        smooth_alpha: 距离 EMA 系数。远距离抖动大，取小值更稳。
        max_jump_m: 匹配时允许的单帧距离跳变上限，超过判为不同目标。
        max_lateral_px: 脚点横向匹配窗口（像素）。
        history_len: 保留给速度估计的 (t, z) 样本数。
        z_min_m/z_max_m: 有效距离区间，区间外的框（如误检在天空）丢弃。
    """

    def __init__(
        self,
        projector: GroundProjector,
        smooth_alpha: float = 0.4,
        max_jump_m: float = 6.0,
        max_lateral_px: float = 80.0,
        history_len: int = 12,
        z_min_m: float = 1.5,
        z_max_m: float = 60.0,
        max_missed: int = 8,
    ) -> None:
        self._proj = projector
        self._alpha = smooth_alpha
        self._max_jump = max_jump_m
        self._max_lat = max_lateral_px
        self._hist_len = history_len
        self._z_range = (z_min_m, z_max_m)
        self._max_missed = max_missed
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)

    def update(self, detections: list[Detection], t_sec: float) -> list[TrackedPedestrian]:
        """输入当帧检测，更新轨迹并返回按距离排序的当帧轨迹列表。"""
        # 1) 脚点反投影，落到有效地面区间内的才参与匹配
        obs: list[tuple[float, float, float, float, tuple[float, float]]] = []
        # 顺序: (u, v, x_m, z_m, conf)
        for det in detections:
            u, v = det.bottom_center
            g = self._proj.unproject(u, v)
            if g is None:
                continue
            x_m, z_m = g
            if not self._z_range[0] <= z_m <= self._z_range[1]:
                continue
            obs.append((u, v, x_m, z_m, det.conf))

        # 2) 最近邻贪心匹配（代价 = 横向像素差 + 加权的距离差）
        pairs: list[tuple[float, int, int]] = []
        for i, o in enumerate(obs):
            for tid, st in self._tracks.items():
                lat = abs(o[0] - st.last_u)
                if lat > self._max_lat:
                    continue
                jump = abs(o[3] - st.z_smooth)
                if jump > self._max_jump:
                    continue
                pairs.append((lat + 2.0 * jump, i, tid))
        pairs.sort(key=lambda p: p[0])

        matched_obs: set[int] = set()
        matched_tracks: set[int] = set()
        for _cost, i, tid in pairs:
            if i in matched_obs or tid in matched_tracks:
                continue
            matched_obs.add(i)
            matched_tracks.add(tid)

        # 3) 更新已匹配轨迹
        for i in matched_obs:
            u, v, x_m, z_m, conf = obs[i]
            tid = next(tid for _c, j, tid in pairs if j == i and tid in matched_tracks)
            st = self._tracks[tid]
            a = self._alpha
            st.z_smooth = a * z_m + (1 - a) * st.z_smooth if st.age else z_m
            st.x_smooth = a * x_m + (1 - a) * st.x_smooth if st.age else x_m
            st.last_u, st.last_v, st.conf = u, v, conf
            st.age += 1
            st.missed = 0
            st.last_t = t_sec
            st.history.append((t_sec, st.z_smooth))
            st.history = st.history[-self._hist_len:]

        # 4) 未匹配的轨迹计数，连续 miss 超限的删除（目标已离开画面）
        dead: list[int] = []
        for tid, st in self._tracks.items():
            if tid in matched_tracks:
                continue
            st.missed += 1
            if st.missed > self._max_missed:
                dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        # 5) 新建未匹配的轨迹
        for i, (u, v, x_m, z_m, conf) in enumerate(obs):
            if i in matched_obs:
                continue
            tid = next(_ids)
            self._tracks[tid] = _TrackState(
                track_id=tid, last_u=u, last_v=v, z_smooth=z_m,
                x_smooth=x_m, conf=conf, age=1,
                history=[(t_sec, z_m)], last_t=t_sec,
            )

        # 5) 输出
        out = [
            TrackedPedestrian(
                track_id=st.track_id,
                distance_m=st.z_smooth,
                lateral_m=st.x_smooth,
                conf=st.conf,
                bottom_uv=(st.last_u, st.last_v),
                age=st.age,
                history=list(st.history),
            )
            for st in self._tracks.values()
        ]
        out.sort(key=lambda p: p.distance_m)
        return out
