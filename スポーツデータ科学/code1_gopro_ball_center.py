"""
code1_gopro_ball_center.py
コード①：目隠しあり（GoPro）条件
- GoPro動画から各フレームのボールを検出
- フレーム中心（=頭部正面方向の代理）からのずれ（距離・方向）を算出
- per-frame CSV / サマリー / （任意）検証用オーバーレイ動画を出力

使い方:
    python code1_gopro_ball_center.py --config config.yaml
    python code1_gopro_ball_center.py --config config.yaml --video data/xxx.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import geom_utils as gu


def parse_args():
    p = argparse.ArgumentParser(description="GoPro(目隠しあり) ボール⇔中心 距離算出")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--video", default=None, help="config を上書きする動画パス")
    p.add_argument("--no-overlay", action="store_true", help="検証動画を出力しない")
    return p.parse_args()


def main():
    import cv2

    args = parse_args()
    cfg = gu.load_config(args.config)
    gcfg = cfg["gopro"]

    video_path = args.video or gcfg["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cx0, cy0 = gu.frame_center(width, height)
    fov_h = gcfg.get("fov_h_deg")
    print(f"[code1] {video_path}  {width}x{height} {fps:.2f}fps  frames={n_frames}")

    # 魚眼歪み補正マップ（任意）
    fmap = None
    fe = gcfg.get("fisheye", {})
    if fe.get("enabled") and fe.get("K") and fe.get("D"):
        fmap = gu.build_fisheye_maps(fe["K"], fe["D"], (width, height))
        print("[code1] 魚眼歪み補正を有効化")

    detector = gu.BallDetector(cfg)
    frame_skip = int(cfg["detect"].get("frame_skip", 0))

    # オーバーレイ動画
    writer = None
    overlay_path = None if args.no_overlay else gcfg.get("overlay_video")
    if overlay_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    rows = []
    prev_xy = None
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if frame_skip > 0 and (idx % (frame_skip + 1) != 0):
            continue

        if fmap is not None:
            frame = cv2.remap(frame, fmap[0], fmap[1],
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

        dets = detector.detect(frame)
        ball = gu.select_ball(dets, prev_xy, width, height, cfg)

        t = idx / fps
        if ball.found:
            dx = ball.cx - cx0
            dy = ball.cy - cy0
            dist_px = math.hypot(dx, dy)
            dist_norm = gu.normalize_by_diagonal(dist_px, width, height)
            dist_deg = gu.px_to_deg(dist_px, width, fov_h)
            prev_xy = (ball.cx, ball.cy)
            rows.append(dict(
                frame_idx=idx, time_sec=t, detected=1,
                ball_x=ball.cx, ball_y=ball.cy, conf=ball.conf,
                dx=dx, dy=dy, dist_px=dist_px, dist_norm=dist_norm, dist_deg=dist_deg,
                interpolated=0,
            ))
        else:
            rows.append(dict(
                frame_idx=idx, time_sec=t, detected=0,
                ball_x=np.nan, ball_y=np.nan, conf=np.nan,
                dx=np.nan, dy=np.nan, dist_px=np.nan, dist_norm=np.nan, dist_deg=np.nan,
                interpolated=0,
            ))

        if writer is not None:
            _draw_overlay(cv2, frame, (cx0, cy0), ball)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)

    # 欠損補間（任意）
    if cfg.get("missing", {}).get("interpolate", False):
        df = _interpolate(df, width, height, fov_h)

    # 出力
    out_csv = gcfg["output_csv"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = _summarize(df, cfg)
    with open(gcfg["output_summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[code1] CSV -> {out_csv}")
    print(f"[code1] summary -> {gcfg['output_summary']}")
    print(f"[code1] 検出率={summary['detection_rate']:.1%}  "
          f"中心保持率={summary['center_hold_rate']:.1%}  "
          f"中央値({summary['hold_metric']})={summary['dist_median']:.2f}")


def _draw_overlay(cv2, frame, center, ball):
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (int(center[0]), int(center[1])), (0, 255, 255),
                   cv2.MARKER_CROSS, 24, 2)
    if ball.found:
        if ball.box:
            x1, y1, x2, y2 = [int(v) for v in ball.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (int(ball.cx), int(ball.cy)), 5, (0, 0, 255), -1)
        cv2.line(frame, (int(center[0]), int(center[1])),
                 (int(ball.cx), int(ball.cy)), (255, 0, 0), 2)


def _interpolate(df, width, height, fov_h):
    """未検出フレームを線形補間し、補間したフレームに interpolated=1 を立てる。"""
    df = df.copy()
    miss = df["detected"] == 0
    df.loc[miss, "interpolated"] = 1
    for col in ["ball_x", "ball_y"]:
        df[col] = df[col].interpolate(limit_direction="both")
    cx0, cy0 = gu.frame_center(width, height)
    recompute = miss
    df.loc[recompute, "dx"] = df.loc[recompute, "ball_x"] - cx0
    df.loc[recompute, "dy"] = df.loc[recompute, "ball_y"] - cy0
    df.loc[recompute, "dist_px"] = np.hypot(df.loc[recompute, "dx"], df.loc[recompute, "dy"])
    df.loc[recompute, "dist_norm"] = df.loc[recompute, "dist_px"] / gu.diagonal(width, height)
    df.loc[recompute, "dist_deg"] = df.loc[recompute, "dist_px"].apply(
        lambda r: gu.px_to_deg(r, width, fov_h))
    return df


def _summarize(df, cfg) -> dict:
    metric = cfg["center_hold"]["metric"]
    thr = cfg["center_hold"]["threshold"]
    col = "dist_deg" if metric == "deg" else "dist_norm"
    valid = df[df["detected"] == 1]
    n = len(df)
    n_det = len(valid)
    series = valid[col].dropna()
    hold_rate = float((series <= thr).mean()) if len(series) else float("nan")
    return dict(
        n_frames=int(n),
        n_detected=int(n_det),
        detection_rate=float(n_det / n) if n else float("nan"),
        hold_metric=metric,
        hold_threshold=float(thr),
        center_hold_rate=hold_rate,
        dist_mean=float(series.mean()) if len(series) else float("nan"),
        dist_median=float(series.median()) if len(series) else float("nan"),
        dist_std=float(series.std()) if len(series) else float("nan"),
        dist_max=float(series.max()) if len(series) else float("nan"),
    )


if __name__ == "__main__":
    main()
