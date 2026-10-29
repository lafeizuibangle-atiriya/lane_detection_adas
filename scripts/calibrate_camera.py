"""相机标定工具：在样例图上恢复俯仰角、航向角与横偏，并写入标定配置。

为什么要跑这一步：项目里所有米制结果（车道宽、曲率半径、横偏、行人距离）
都建立在相机几何上。以前这些参数是手调的，结果是同一段路在不同纵深上
算出的车道宽从 2.4 m 漂到 4.65 m。这里改成从图像本身解出来。

焦距由镜头标称水平视场角给出，俯仰角与航向角从两条车道线的消失点求——
单帧直线车道无法同时确定焦距和俯仰角，原因见 adas/lane/calibrate.py 的注释。

用法：
    python scripts/calibrate_camera.py                     # 跑全部样例图，只打印
    python scripts/calibrate_camera.py --write             # 额外写入 configs/calibrated.yaml
    python scripts/calibrate_camera.py data/samples/images/solidWhiteRight.jpg
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas.core.config import CameraConfig, load_config  # noqa: E402
from adas.lane.binarize import binarize  # noqa: E402
from adas.lane.calibrate import CalibResult, calibrate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_one(path: Path, cam: CameraConfig, cfg) -> CalibResult | None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  {path.name}: 读不到图")
        return None
    mask = binarize(img, cfg.threshold)
    local = cam.model_copy(update={"width": img.shape[1], "height": img.shape[0]})
    return calibrate(mask, camera=local)


def run_video(path: Path, cam: CameraConfig, cfg, stride: int = 5) -> list[CalibResult]:
    """扫描视频，每隔 stride 帧标定一次，返回成功结果。

    单帧标定在弯道上偏低：切线外推的消失点随弯向系统偏移，
    但左右弯的偏移方向相反，取多帧中位数能把它抵掉大半。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  {path.name}: 打不开视频")
        return []
    out: list[CalibResult] = []
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % stride == 0:
            mask = binarize(frame, cfg.threshold)
            r = calibrate(mask, camera=cam)
            if r is not None:
                out.append(r)
        k += 1
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="留空则跑 data/samples/images 下全部 jpg")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--write", action="store_true", help="把中位数结果写入 configs/calibrated.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam = cfg.camera
    paths = [Path(p) for p in args.images] or sorted((ROOT / "data/samples/images").glob("*.jpg"))
    if not paths:
        print("没有找到样例图")
        return

    print(f"标称：hfov={cam.hfov_deg}°（作为焦距输入） h={cam.height_m} m "
          f"lane_width={cfg.road.lane_width_m} m")
    print(f"{'图像':<26}{'pitch°':>8}{'yaw°':>8}{'横偏m':>8}{'残差px':>8}{'尺度比':>8}{'像素':>7}")
    ok: list[CalibResult] = []
    for p in paths:
        r = run_one(p, cam, cfg)
        if r is None:
            print(f"{p.name:<26}{'标定失败':>8}")
            continue
        ok.append(r)
        flag = "" if r.ok else "  ← 尺度比偏离，视场角可能不对"
        print(f"{p.name:<26}{r.pitch_deg:8.2f}{r.yaw_deg:8.2f}{r.offset_m:8.3f}"
              f"{r.residual_px:8.2f}{r.scale_ratio:8.3f}{r.n_pixels:7d}{flag}")

    # 视频多帧统计：单帧结果受弯道影响，中位数更可靠
    for vp in sorted((ROOT / "data/samples/videos").glob("*.mp4")):
        rs = run_video(vp, cam, cfg)
        if not rs:
            print(f"视频 {vp.name}: 无成功帧")
            continue
        pitches = sorted(r.pitch_deg for r in rs)
        med = pitches[len(pitches) // 2]
        q1, q3 = pitches[len(pitches) // 4], pitches[(3 * len(pitches)) // 4]
        ratios = [r.scale_ratio for r in rs if abs(r.scale_ratio - 1.0) < 0.15]
        print(f"视频 {vp.name}: {len(rs)} 帧成功，pitch 中位 {med:.2f}° "
              f"(四分位 {q1:.2f}~{q3:.2f})")
        if ratios:
            rm = ratios[len(ratios) // 2]
            print(f"  尺度比中位 {rm:.3f}（{len(ratios)}/{len(rs)} 帧在 ±15% 内）")
            ok.append(CalibResult(pitch_deg=med, focal_px=cam.focal_px, hfov_deg=cam.hfov_deg,
                                  offset_m=0.0, yaw_deg=0.0, residual_px=0.0,
                                  scale_ratio=rm, n_pixels=0))

    if not ok:
        print("\n全部失败：检查阈值或者确认图里有清晰的双车道线。")
        return

    pitch = statistics.median(r.pitch_deg for r in ok)
    yaw = statistics.median(r.yaw_deg for r in ok)
    ratio = statistics.median(r.scale_ratio for r in ok)
    res = statistics.median(r.residual_px for r in ok)
    print(f"\n{len(ok)}/{len(paths)} 张成功；中位 pitch={pitch:.2f}° yaw={yaw:.2f}° "
          f"残差={res:.2f}px 尺度比={ratio:.3f}")
    if abs(ratio - 1.0) > 0.12:
        print("尺度比偏离 1 超过 12%：标称视场角或车道宽与图像不符，"
              "先用 --write 之外的方式核对镜头规格，别直接拿结果做米制结论。")

    if args.write:
        out = ROOT / "configs/calibrated.yaml"
        out.write_text(
            "# 由 scripts/calibrate_camera.py 生成，覆盖 default.yaml 的 camera 段\n"
            "camera:\n"
            f"  width: {cam.width}\n"
            f"  height: {cam.height}\n"
            f"  height_m: {cam.height_m}\n"
            f"  pitch_deg: {round(pitch, 3)}\n"
            f"  hfov_deg: {cam.hfov_deg}\n",
            encoding="utf-8",
        )
        print(f"已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
