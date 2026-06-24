"""
code2_tobii_ball_gaze.py
コード②：目隠し無し（Tobii）条件
- Tobiiシーン動画から各フレームのボールを検出
- フレーム中心 と 視線点(gaze) それぞれからの距離を算出
- AOI判定（gazeがボールbbox内にあるか）
- per-frame CSV / サマリー / （任意）オーバーレイ動画を出力

使い方:
    python code2_tobii_ball_gaze.py --config config.yaml
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
    p = argparse.ArgumentParser(description="Tobii(目隠し無し) ボール認識 & gaze距離")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--video", default=None)
    p.add_argument("--gaze", default=None)
    p.add_argument("--no-overlay", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------
# gaze データ読み込み・整列
# ------------------------------------------------------------
def load_gaze(tcfg: dict) -> pd.DataFrame:
    """
    Tobii の gaze データ（TSV/CSV）を読み込み、共通形式 [t_sec, gx, gy, valid] に整形する。
    gx, gy は 0-1 正規化座標。列名は config の gaze_columns で指定。
    """
    path = tcfg["gaze_path"]
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    raw = pd.read_csv(path, sep=sep)

    cols = tcfg["gaze_columns"]
    t = raw[cols["timestamp"]].astype(float).to_numpy()
    # 時刻単位を秒へ
    unit = tcfg.get("gaze_time_unit", "s")
    factor = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[unit]
    t = t * factor
    t = t - t[0]                                   # 0始まりに
    t = t + float(tcfg.get("sync_offset_sec", 0.0))  # 同期オフセット

    gx = raw[cols["x"]].astype(float).to_numpy()
    gy = raw[cols["y"]].astype(float).to_numpy()

    # 有効性
    if cols.get("validity") and cols["validity"] in raw.columns:
        vraw = raw[cols["validity"]]
        valid_val = tcfg.get("gaze_valid_value", 1)
        valid = (vraw == valid_val).to_numpy()
    else:
        valid = np.isfinite(gx) & np.isfinite(gy)

    g = pd.DataFrame(dict(t_sec=t, gx=gx, gy=gy, valid=valid.astype(bool)))
    return g.sort_values("t_sec").reset_index(drop=True)


def gaze_at_time(gaze: pd.DataFrame, t: float, mode: str, frame_dt: float):
    """指定時刻のgazeを取得。nearest=最近傍 / mean=±frame_dt/2区間の平均。"""
    if mode == "mean":
        lo, hi = t - frame_dt / 2, t + frame_dt / 2
        seg = gaze[(gaze["t_sec"] >= lo) & (gaze["t_sec"] < hi) & (gaze["valid"])]
        if len(seg) == 0:
            return np.nan, np.nan, False
        return float(seg["gx"].mean()), float(seg["gy"].mean()), True
    # nearest
    i = int((gaze["t_sec"] - t).abs().values.argmin())
    row = gaze.iloc[i]
    # フレーム間隔より離れていたら欠損扱い
    if abs(row["t_sec"] - t) > frame_dt:
        return np.nan, np.nan, False
    return float(row["gx"]), float(row["gy"]), bool(row["valid"])


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main():
    import cv2

    args = parse_args()
    cfg = gu.load_config(args.config)
    tcfg = cfg["tobii"]
    if args.video:
        tcfg["video_path"] = args.video
    if args.gaze:
        tcfg["gaze_path"] = args.gaze

    for k in ["video_path", "gaze_path"]:
        if not os.path.exists(tcfg[k]):
            raise FileNotFoundError(f"見つかりません: {tcfg[k]}")

    cap = cv2.VideoCapture(tcfg["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {tcfg['video_path']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cx0, cy0 = gu.frame_center(width, height)
    fov_h = tcfg.get("fov_h_deg")
    frame_dt = 1.0 / fps
    print(f"[code2] {tcfg['video_path']}  {width}x{height} {fps:.2f}fps frames={n_frames}")

    gaze = load_gaze(tcfg)
    print(f"[code2] gaze {len(gaze)} samples  有効率={gaze['valid'].mean():.1%}")
    resample = tcfg.get("gaze_resample", "nearest")

    detector = gu.BallDetector(cfg)
    frame_skip = int(cfg["detect"].get("frame_skip", 0))

    writer = None
    overlay_path = None if args.no_overlay else tcfg.get("overlay_video")
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

        t = idx / fps
        dets = detector.detect(frame)
        ball = gu.select_ball(dets, prev_xy, width, height, cfg)

        # gaze（正規化 -> ピクセル）
        gx_n, gy_n, gvalid = gaze_at_time(gaze, t, resample, frame_dt)
        gx_px = gx_n * width if np.isfinite(gx_n) else np.nan
        gy_px = gy_n * height if np.isfinite(gy_n) else np.nan

        rec = dict(frame_idx=idx, time_sec=t)
        if ball.found:
            prev_xy = (ball.cx, ball.cy)
            # ボール⇔中心
            dxc, dyc = ball.cx - cx0, ball.cy - cy0
            dist_c = math.hypot(dxc, dyc)
            rec.update(detected=1, ball_x=ball.cx, ball_y=ball.cy, conf=ball.conf,
                       dx_center=dxc, dy_center=dyc,
                       dist_ball_center_px=dist_c,
                       dist_ball_center_norm=gu.normalize_by_diagonal(dist_c, width, height),
                       dist_ball_center_deg=gu.px_to_deg(dist_c, width, fov_h))
            # ボール⇔gaze
            if gvalid and np.isfinite(gx_px):
                dist_g = math.hypot(ball.cx - gx_px, ball.cy - gy_px)
                on_ball = _point_in_box((gx_px, gy_px), ball.box)
                rec.update(dist_ball_gaze_px=dist_g,
                           dist_ball_gaze_norm=gu.normalize_by_diagonal(dist_g, width, height),
                           dist_ball_gaze_deg=gu.px_to_deg(dist_g, width, fov_h),
                           gaze_on_ball=int(on_ball))
            else:
                rec.update(dist_ball_gaze_px=np.nan, dist_ball_gaze_norm=np.nan,
                           dist_ball_gaze_deg=np.nan, gaze_on_ball=np.nan)
        else:
            rec.update(detected=0, ball_x=np.nan, ball_y=np.nan, conf=np.nan,
                       dx_center=np.nan, dy_center=np.nan,
                       dist_ball_center_px=np.nan, dist_ball_center_norm=np.nan,
                       dist_ball_center_deg=np.nan,
                       dist_ball_gaze_px=np.nan, dist_ball_gaze_norm=np.nan,
                       dist_ball_gaze_deg=np.nan, gaze_on_ball=np.nan)

        rec.update(gaze_x=gx_px, gaze_y=gy_px, gaze_valid=int(bool(gvalid)))
        rows.append(rec)

        if writer is not None:
            _draw_overlay(cv2, frame, (cx0, cy0), ball,
                          (gx_px, gy_px) if gvalid else None)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    out_csv = tcfg["output_csv"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = _summarize(df, cfg)
    with open(tcfg["output_summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[code2] CSV -> {out_csv}")
    print(f"[code2] 検出率={summary['detection_rate']:.1%}  "
          f"gaze有効率={summary['gaze_valid_rate']:.1%}  "
          f"AOIヒット率={summary['aoi_hit_rate']:.1%}")


def _point_in_box(p, box):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return (x1 <= p[0] <= x2) and (y1 <= p[1] <= y2)


def _draw_overlay(cv2, frame, center, ball, gaze_px):
    cv2.drawMarker(frame, (int(center[0]), int(center[1])), (0, 255, 255),
                   cv2.MARKER_CROSS, 24, 2)
    if ball.found:
        if ball.box:
            x1, y1, x2, y2 = [int(v) for v in ball.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (int(ball.cx), int(ball.cy)), 5, (0, 0, 255), -1)
    if gaze_px is not None and np.isfinite(gaze_px[0]):
        cv2.circle(frame, (int(gaze_px[0]), int(gaze_px[1])), 10, (0, 255, 0), 2)


def _summarize(df, cfg) -> dict:
    metric = cfg["center_hold"]["metric"]
    thr = cfg["center_hold"]["threshold"]
    ccol = "dist_ball_center_deg" if metric == "deg" else "dist_ball_center_norm"
    gcol = "dist_ball_gaze_deg" if metric == "deg" else "dist_ball_gaze_norm"
    n = len(df)
    det = df[df["detected"] == 1]
    cser = det[ccol].dropna()
    gser = det[gcol].dropna()
    aoi = det["gaze_on_ball"].dropna()
    return dict(
        n_frames=int(n),
        detection_rate=float((df["detected"] == 1).mean()) if n else float("nan"),
        gaze_valid_rate=float((df["gaze_valid"] == 1).mean()) if n else float("nan"),
        hold_metric=metric,
        ball_center_median=float(cser.median()) if len(cser) else float("nan"),
        ball_center_std=float(cser.std()) if len(cser) else float("nan"),
        ball_gaze_median=float(gser.median()) if len(gser) else float("nan"),
        ball_gaze_std=float(gser.std()) if len(gser) else float("nan"),
        center_hold_rate=float((cser <= thr).mean()) if len(cser) else float("nan"),
        aoi_hit_rate=float(aoi.mean()) if len(aoi) else float("nan"),
    )


if __name__ == "__main__":
    main()
