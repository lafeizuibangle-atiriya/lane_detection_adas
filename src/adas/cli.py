"""统一命令行入口。

    python -m adas run <图片|视频> [-o output] [--detector yolo|hog|none]
    python -m adas baseline <视频> [...]          # v0 基线，同口径输出
    python -m adas eval <视频> [...]              # v0/v1 对比表

设计原则：配置驱动（configs/default.yaml），输出集中到 --output 目录
（标注媒体 + results.json），日志带阶段耗时，错误给明确提示而不是裸
Traceback。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from adas.core.config import AppConfig, load_config
from adas.obstacle.pedestrian import make_detector

logger = logging.getLogger("adas")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_pipeline(cfg: AppConfig, detector_kind: str, weights: str):
    """装配管线。detector=none 时只跑车道分支。"""
    from adas.pipeline import AdasPipeline

    det = None
    if detector_kind != "none":
        kwargs = {"weights": weights} if detector_kind == "yolo" else {}
        det = make_detector(detector_kind, **kwargs)
    return AdasPipeline(cfg, detector=det)


def open_input(path: Path):
    """图片或视频统一打开，返回 (kind, iterator, fps, size)。"""
    img = cv2.imread(str(path))
    if img is not None:
        return "image", iter([(img, 0.0)]), 0.0, (img.shape[1], img.shape[0])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"读不了 {path}：既不是可识别的图片，也不是可解码的视频")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return "video", cap, fps, size


def run_source(pipe, source, kind, fps, out_dir: Path, stem: str,
               max_frames: int = 0) -> dict:
    """跑完一个输入源，落盘标注媒体与 results.json，返回汇总统计。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    records: list[dict] = []
    t_start = time.perf_counter()
    n = 0

    if kind == "image":
        img, t = next(source)
        r = pipe.process_frame(img, t, 0)
        cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), r.annotated)
        records.append(r.to_json())
        n = 1
    else:
        cap = source
        out_path = out_dir / f"{stem}_annotated.mp4"
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                # 按首帧实际尺寸开 writer（灰度/异常尺寸输入在管线里已归一）
                probe = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (pipe.cfg.camera.width, pipe.cfg.camera.height),
                )
            r = pipe.process_frame(frame, n / fps, n)
            writer.write(r.annotated)
            records.append(r.to_json())
            n += 1
            if n % 100 == 0:
                logger.info("进度 %d 帧, 最高告警 %s", n, r.max_level.label)
            if max_frames and n >= max_frames:
                break
        if writer is not None:
            writer.release()
        cap.release()

    elapsed = time.perf_counter() - t_start
    timings = pipe.timings_ms
    stages = {k: round(v / max(n, 1), 1) for k, v in timings.items()}

    # 汇总统计
    lane_found = sum(1 for r in records if r["lane"]["found"])
    offsets = [r["lane"]["offset_m"] for r in records if r["lane"]["offset_m"] is not None]
    peak = max((r["max_level"] for r in records),
               key=lambda lv: ["SAFE", "HINT", "WARNING", "EMERGENCY"].index(lv))
    summary = {
        "input_stem": stem,
        "frames": n,
        "wall_time_s": round(elapsed, 1),
        "fps": round(n / max(elapsed, 1e-6), 1),
        "lane_found_rate": round(lane_found / max(n, 1), 3),
        "offset_median_m": round(float(np.median(offsets)), 3) if offsets else None,
        "offset_jitter_m": round(float(np.std(np.diff(offsets))), 3)
        if len(offsets) > 2 else None,
        "peak_level": peak,
        "stage_ms_per_frame": stages,
    }
    with (out_dir / f"{stem}_results.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": records}, f, ensure_ascii=False, indent=1)

    logger.info("完成 %s: %d 帧, %.1f FPS, 车道检出率 %.0f%%, 最高告警 %s",
                stem, n, summary["fps"], summary["lane_found_rate"] * 100, peak)
    logger.info("各阶段平均耗时 ms/帧: %s", stages)
    return summary


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    src = Path(args.input)
    if not src.exists():
        logger.error("输入不存在: %s", src)
        return 1
    try:
        pipe = build_pipeline(cfg, args.detector, args.weights)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    kind, source, fps, _size = open_input(src)
    summary = run_source(pipe, source, kind, fps, Path(args.output), src.stem,
                         max_frames=args.max_frames)
    if summary["offset_median_m"] is not None:
        logger.info("中心偏移中位 %+.3f m (抖动 %.3f m)",
                    summary["offset_median_m"], summary["offset_jitter_m"] or -1)
    return 0


def cmd_baseline(args) -> int:
    """v0 基线：同一批输入跑 Hough 方法，输出同口径 summary。"""
    from adas.baseline import BaselineHough

    out_dir = Path(args.output)
    rows = []
    for p in (Path(x) for x in args.inputs):
        if not p.exists():
            logger.error("输入不存在: %s", p)
            continue
        kind, source, fps, _ = open_input(p)
        base = BaselineHough()
        offsets, found, elapsed = [], 0, 0.0
        n = 0
        frames = []
        while True:
            if kind == "image":
                pair = next(source, None)
                if pair is None:
                    break
                img = pair[0]
            else:
                ok, img = source.read()
                if not ok:
                    break
            r = base.process_frame(img)
            offsets.append(r["offset_m"])
            found += int(r["found"])
            elapsed += r["elapsed_ms"]
            n += 1
            if args.save_frames and n in tuple(args.save_frames):
                frames.append((n, base.draw(img, r)))
            if args.max_frames and n >= args.max_frames:
                break
        if kind == "video":
            source.release()
        offs = [o for o in offsets if o is not None]
        jit = float(np.std(np.diff(offs))) if len(offs) > 2 else None
        row = {
            "video": p.stem, "method": "v0_hough", "frames": n,
            "found_rate": round(found / max(n, 1), 3),
            "offset_median_m": round(float(np.median(offs)), 3) if offs else None,
            "offset_jitter_m": round(jit, 3) if jit is not None else None,
            "ms_per_frame": round(elapsed / max(n, 1), 1),
        }
        rows.append(row)
        logger.info("v0 %s: 检出率 %.0f%%, 抖动 %s m, %.1f ms/帧",
                    p.stem, row["found_rate"] * 100, row["offset_jitter_m"], row["ms_per_frame"])
        for n_, drawn in frames:
            cv2.imwrite(str(out_dir / f"{p.stem}_v0_f{n_}.jpg"), drawn)
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_eval(args) -> int:
    """v0/v1 同口径对比：车道检出率、中心偏移抖动、单帧耗时。"""
    cfg = load_config(args.config)
    pipe = build_pipeline(cfg, "none", "")
    from adas.baseline import BaselineHough

    rows = []
    for p in (Path(x) for x in args.inputs):
        if not p.exists():
            logger.error("输入不存在: %s", p)
            continue
        # v1
        kind, source, fps, _ = open_input(p)
        offsets, found, n = [], 0, 0
        t0 = time.perf_counter()
        while True:
            if kind == "image":
                pair = next(source, None)
                if pair is None:
                    break
                img = pair[0]
            else:
                ok, img = source.read()
                if not ok:
                    break
            r = pipe.process_frame(img, n / fps, n)
            if r.lane_offset_m is not None:
                offsets.append(r.lane_offset_m)
            found += int(r.lane_found)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        v1_time = time.perf_counter() - t0
        if kind == "video":
            source.release()
        v1_offs = offsets
        v1_row = {
            "video": p.stem, "method": "v1_perspective",
            "found_rate": round(found / max(n, 1), 3),
            "offset_median_m": round(float(np.median(v1_offs)), 3) if v1_offs else None,
            "offset_jitter_m": round(float(np.std(np.diff(v1_offs))), 3)
            if len(v1_offs) > 2 else None,
            "ms_per_frame": round(v1_time * 1000 / max(n, 1), 1),
        }
        # v0
        base = BaselineHough()
        kind, source, fps, _ = open_input(p)
        offsets, found, elapsed, n = [], 0, 0.0, 0
        while True:
            if kind == "image":
                pair = next(source, None)
                if pair is None:
                    break
                img = pair[0]
            else:
                ok, img = source.read()
                if not ok:
                    break
            r = base.process_frame(img)
            if r["offset_m"] is not None:
                offsets.append(r["offset_m"])
            found += int(r["found"])
            elapsed += r["elapsed_ms"]
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        if kind == "video":
            source.release()
        v0_row = {
            "video": p.stem, "method": "v0_hough",
            "found_rate": round(found / max(n, 1), 3),
            "offset_median_m": round(float(np.median(offsets)), 3) if offsets else None,
            "offset_jitter_m": round(float(np.std(np.diff(offsets))), 3)
            if len(offsets) > 2 else None,
            "ms_per_frame": round(elapsed / max(n, 1), 1),
        }
        rows += [v1_row, v0_row]
        logger.info("%s: v1 抖动 %s / v0 抖动 %s", p.stem,
                    v1_row["offset_jitter_m"], v0_row["offset_jitter_m"])

    out = Path(args.output) / "eval_compare.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    # 控制台表格
    print(f"\n{'video':<22}{'method':<16}{'found%':>8}{'offset中位':>11}{'抖动m':>8}{'ms/帧':>8}")
    for r in rows:
        print(f"{r['video']:<22}{r['method']:<16}{r['found_rate']*100:>7.0f}%"
              f"{r['offset_median_m'] if r['offset_median_m'] is not None else '--':>11}"
              f"{r['offset_jitter_m'] if r['offset_jitter_m'] is not None else '--':>8}"
              f"{r['ms_per_frame']:>8.1f}")
    logger.info("对比数据已写入 %s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adas", description="单目 ADAS 原型：车道线 + 行人 TTC")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="跑图片或视频，输出标注媒体与 JSON")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="output")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--detector", choices=["yolo", "hog", "none"], default="yolo")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--max-frames", type=int, default=0, help="视频最多处理多少帧，0 = 全部")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("baseline", help="v0 Hough 基线")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", default="output")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-frames", type=int, nargs="*", default=[],
                   help="额外保存这些帧号的 v0 标注图")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("eval", help="v0/v1 同口径对比表")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", default="output")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("中断")
        return 130
    except Exception:
        logger.exception("运行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
