"""
code3_compare_visualize.py
コード③：2条件の比較・分析補助 & 可視化
- code1（目隠しあり）/ code2（目隠し無し）の per-frame CSV を読み込む
- 共通指標（視野角[度] or 対角正規化）で揃えて比較
- 記述統計 / 箱ひげ図 / 時系列 / ヒストグラム / 2Dヒートマップ
- 視覚あり条件の内部比較（ボール⇔中心 vs ボール⇔gaze）
- （補助）Mann-Whitney U 検定（注意書き付き）

使い方:
    python code3_compare_visualize.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import geom_utils as gu


def parse_args():
    p = argparse.ArgumentParser(description="2条件の比較・可視化")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def _center_col(metric: str) -> str:
    return "dist_ball_center_deg" if metric == "deg" else "dist_ball_center_norm"


def load_condition(csv_path: str, label: str, metric: str, kind: str,
                   trim) -> pd.DataFrame:
    """
    1条件のCSVを読み込み、共通列 dist_center（+ vision なら dist_gaze）を作る。
    kind: "blindfold"(code1) or "vision"(code2)
    """
    df = pd.read_csv(csv_path)
    df = df[df["detected"] == 1].copy()

    # code1 は dist_deg / dist_norm、code2 は dist_ball_center_deg/norm。両対応にする。
    if kind == "blindfold":
        src = "dist_deg" if metric == "deg" else "dist_norm"
        df["dist_center"] = df[src]
        df["dist_gaze"] = np.nan
    else:
        df["dist_center"] = df[_center_col(metric)]
        gcol = "dist_ball_gaze_deg" if metric == "deg" else "dist_ball_gaze_norm"
        df["dist_gaze"] = df[gcol] if gcol in df.columns else np.nan

    # 比較区間トリミング
    if trim:
        s, e = trim.get("start_sec"), trim.get("end_sec")
        if s is not None:
            df = df[df["time_sec"] >= s]
        if e is not None:
            df = df[df["time_sec"] <= e]

    df["condition"] = label
    df["kind"] = kind
    keep = ["frame_idx", "time_sec", "condition", "kind",
            "ball_x", "ball_y", "dist_center", "dist_gaze"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def descriptive_stats(df_all: pd.DataFrame, metric: str, thr: float) -> pd.DataFrame:
    out = []
    for cond, g in df_all.groupby("condition"):
        s = g["dist_center"].dropna()
        out.append(dict(
            condition=cond, metric=metric, n=len(s),
            mean=s.mean(), median=s.median(), std=s.std(),
            q25=s.quantile(0.25), q75=s.quantile(0.75),
            iqr=s.quantile(0.75) - s.quantile(0.25),
            center_hold_rate=float((s <= thr).mean()) if len(s) else np.nan,
        ))
    return pd.DataFrame(out)


def stat_test(df_all: pd.DataFrame):
    """2条件の dist_center 分布を Mann-Whitney U で比較。"""
    from scipy.stats import mannwhitneyu
    conds = df_all["condition"].unique()
    if len(conds) != 2:
        return None
    a = df_all[df_all["condition"] == conds[0]]["dist_center"].dropna()
    b = df_all[df_all["condition"] == conds[1]]["dist_center"].dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return dict(group_a=str(conds[0]), group_b=str(conds[1]),
                n_a=int(len(a)), n_b=int(len(b)),
                U=float(u), p_value=float(p),
                note=("※フレーム単位の観測は時系列的に独立でないため、p値は厳密な"
                      "仮説検定ではなく分布差の目安として解釈すること。試計測・少数例。"))


# ------------------------------------------------------------
# 可視化
# ------------------------------------------------------------
def plot_box(df_all, metric, out_dir, colors):
    import matplotlib.pyplot as plt
    conds = list(df_all["condition"].unique())
    data = [df_all[df_all["condition"] == c]["dist_center"].dropna() for c in conds]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = _boxplot(ax, data, conds, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(colors.get(c, "#cccccc"))
        patch.set_alpha(0.7)
    ax.set_ylabel(f"ボール⇔中心 距離 ({_unit(metric)})")
    ax.set_title("条件別 ボール⇔中心 距離の分布")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "box_ball_center")


def plot_timeseries(df_all, metric, out_dir, colors):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for c, g in df_all.groupby("condition"):
        g = g.sort_values("time_sec")
        ax.plot(g["time_sec"], g["dist_center"], label=c,
                color=colors.get(c), lw=1.0, alpha=0.8)
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel(f"ボール⇔中心 距離 ({_unit(metric)})")
    ax.set_title("ボール⇔中心 距離の時系列")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "timeseries_ball_center")


def plot_hist(df_all, metric, out_dir, colors):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, g in df_all.groupby("condition"):
        s = g["dist_center"].dropna()
        ax.hist(s, bins=30, alpha=0.5, density=True, label=c, color=colors.get(c))
    ax.set_xlabel(f"ボール⇔中心 距離 ({_unit(metric)})")
    ax.set_ylabel("密度")
    ax.set_title("ボール⇔中心 距離のヒストグラム")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "hist_ball_center")


def plot_heatmap(df_all, out_dir):
    """フレーム上のボール位置の2Dヒートマップ（条件別、正規化座標0-1）。"""
    import matplotlib.pyplot as plt
    conds = list(df_all["condition"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 5), squeeze=False)
    for ax, c in zip(axes[0], conds):
        g = df_all[df_all["condition"] == c]
        if {"ball_x", "ball_y"}.issubset(g.columns):
            # 解像度が条件で違うため 0-1 正規化（各条件の最大値で割る簡易版）
            x = g["ball_x"] / g["ball_x"].max()
            y = g["ball_y"] / g["ball_y"].max()
            ax.hist2d(x, y, bins=30, range=[[0, 1], [0, 1]], cmap="jet")
        ax.plot(0.5, 0.5, "w+", markersize=16, markeredgewidth=2)  # フレーム中心
        ax.set_title(f"ボール位置分布: {c}")
        ax.set_xlabel("x (正規化)")
        ax.set_ylabel("y (正規化)")
        ax.invert_yaxis()
    _save(fig, out_dir, "heatmap_ball_position")


def plot_gaze_vs_center(df_all, metric, out_dir):
    """視覚あり条件のみ：ボール⇔中心 と ボール⇔gaze を比較。"""
    import matplotlib.pyplot as plt
    vis = df_all[(df_all["kind"] == "vision")]
    if vis.empty or vis["dist_gaze"].dropna().empty:
        return
    center = vis["dist_center"].dropna()
    gaze = vis["dist_gaze"].dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = _boxplot(ax, [center, gaze], ["ボール⇔中心\n(頭部方向)", "ボール⇔gaze\n(視線)"],
                  showmeans=True, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#4C72B0", "#55A868"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_ylabel(f"距離 ({_unit(metric)})")
    ax.set_title("視覚あり条件：頭部方向 vs 視線（どちらがボールに近いか）")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "vision_gaze_vs_center")


def _boxplot(ax, data, tick_labels, **kwargs):
    """matplotlib のバージョン差（labels -> tick_labels）を吸収する。"""
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=tick_labels, **kwargs)


def _unit(metric):
    return "度" if metric == "deg" else "正規化"


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[code3] 図 -> {path}")


# ------------------------------------------------------------
# レポート
# ------------------------------------------------------------
def write_report(out_dir, stats_df, test_res, metric, labels):
    lines = ["# 比較分析レポート（自動生成）", "",
             f"共通指標: ボール⇔中心 距離（{_unit(metric)}）", "",
             "## 記述統計", "", stats_df.round(3).to_markdown(index=False), ""]
    if test_res:
        lines += ["## 分布差の検定（Mann-Whitney U）", "",
                  f"- {test_res['group_a']} (n={test_res['n_a']}) vs "
                  f"{test_res['group_b']} (n={test_res['n_b']})",
                  f"- U = {test_res['U']:.1f}, p = {test_res['p_value']:.4g}",
                  f"- {test_res['note']}", ""]
    lines += ["## 解釈の注意",
              "- 機材（GoPro / Tobii）でFOV・解像度が異なるため、視野角換算で揃えた指標で比較している。",
              "- フレーム単位の観測は時間的に独立でないため、検定p値は目安。",
              "- 試計測・少数例のため、結論は探索的な傾向として扱うこと。"]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[code3] レポート -> {path}")


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main():
    args = parse_args()
    cfg = gu.load_config(args.config)
    ccfg = cfg["compare"]
    metric = ccfg.get("metric", "deg")
    thr = cfg["center_hold"]["threshold"]
    out_dir = ccfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    gu.setup_japanese_font()

    labels = ccfg["labels"]
    trim = ccfg.get("trim")
    df_b = load_condition(ccfg["code1_csv"], labels["blindfold"], metric, "blindfold", trim)
    df_v = load_condition(ccfg["code2_csv"], labels["vision"], metric, "vision", trim)
    df_all = pd.concat([df_b, df_v], ignore_index=True)
    print(f"[code3] frames: blindfold={len(df_b)}  vision={len(df_v)}")

    # 記述統計
    stats_df = descriptive_stats(df_all, metric, thr)
    stats_df.to_csv(os.path.join(out_dir, "descriptive_stats.csv"),
                    index=False, encoding="utf-8-sig")
    print("[code3] 記述統計:")
    print(stats_df.round(3).to_string(index=False))

    # 検定（任意）
    test_res = None
    if ccfg.get("run_stat_test", True):
        test_res = stat_test(df_all)

    # 配色
    colors = {labels["blindfold"]: "#C44E52", labels["vision"]: "#4C72B0"}

    # 可視化
    plot_box(df_all, metric, out_dir, colors)
    plot_timeseries(df_all, metric, out_dir, colors)
    plot_hist(df_all, metric, out_dir, colors)
    plot_heatmap(df_all, out_dir)
    plot_gaze_vs_center(df_all, metric, out_dir)

    # レポート
    write_report(out_dir, stats_df, test_res, metric, labels)
    print(f"[code3] 完了 -> {out_dir}")


if __name__ == "__main__":
    main()
