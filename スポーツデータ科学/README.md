# 視覚情報の有無によるボール捕捉特性の比較分析

スポーツデータ科学 第7回宿題。フットサルのパス回しで「目隠しあり（GoPro＝頭部方向のみ）」と
「目隠し無し（Tobii＝視線あり）」を比較し、視覚があるとボールを視野中心に保持しやすくなるかを調べる。

## 構成

| ファイル | 役割 |
|----------|------|
| `config.yaml` | 全コード共通の設定（パス・閾値・FOV・gaze列名など） |
| `geom_utils.py` | 共通ユーティリティ（座標変換・距離・ボール検出器・日本語フォント） |
| `code1_gopro_ball_center.py` | ①目隠しあり：ボール検出 → フレーム中心からの距離 |
| `code2_tobii_ball_gaze.py` | ②目隠し無し：ボール検出 → 中心 / 視線(gaze) からの距離 |
| `code3_compare_visualize.py` | ③2条件の比較・統計・可視化 |

## セットアップ

```bash
pip install -r requirements.txt
```

YOLOの既存フットサルカスタムモデルを `models/ball_yolo.pt` に置く。
無い場合は自動で **HSV色検出にフォールバック**する（`config.yaml` の `color_fallback` を要調整）。

## 実行

```bash
# ① 目隠しあり（GoPro）
python code1_gopro_ball_center.py --config config.yaml

# ② 目隠し無し（Tobii）
python code2_tobii_ball_gaze.py --config config.yaml

# ③ 比較・可視化（①②のCSVを読み込む）
python code3_compare_visualize.py --config config.yaml
```

出力は `outputs/` に per-frame CSV・サマリーJSON・検証動画、`outputs/compare/` に図とレポート。

## 設定の要点

- **`detect.model_path`** … YOLO重み。`ball` クラス名を自動検出（`ball_class_id` で明示も可）。
- **`gopro.fov_h_deg` / `tobii.fov_h_deg`** … 各機材の水平視野角。**距離を「度」に揃える肝**なので実測値に合わせる。
  - GoProは設定（Wide/Linear等）でFOVが変わる。魚眼歪みが強い場合は `gopro.fisheye` でキャリブレーション値(K,D)を入れて補正。
- **`center_hold.metric`** … `deg`（視野角）推奨。`norm`（対角正規化）も可。`threshold` が「中心保持」の許容角。
- **`tobii.gaze_columns` / `gaze_time_unit`** … gazeファイルの列名と時刻単位。
  - ⚠️ **時刻単位の指定ミスに注意**。Tobiiのエクスポートが ms なのに `s` のままだと、整列が全部ズレてgaze有効率がほぼ0になる。`s` / `ms` / `us` を実データに合わせること。
  - 動画とgazeの開始ずれは `tobii.sync_offset_sec` で補正。

## 出力カラム（主なもの）

- ①: `dist_px / dist_norm / dist_deg`（ボール⇔フレーム中心）
- ②: `dist_ball_center_*`（ボール⇔中心）, `dist_ball_gaze_*`（ボール⇔視線）, `gaze_on_ball`（AOIヒット）
- ③: `descriptive_stats.csv`（条件別 mean/median/std/IQR/中心保持率）, 各種PNG, `report.md`

## 分析の注意（レポートにも明記される）

- 機材差（FOV・解像度）があるため、**視野角換算で揃えた指標**で比較する。ピクセル距離の直接比較はしない。
- フレーム単位の観測は時間的に独立でないため、検定のp値は**分布差の目安**。
- 試計測・少数例なので、結論は探索的な傾向として扱う。

## 動作確認状況

`geom_utils`（座標・視野角変換）、`code1`（動画→検出→距離→CSV）、`code2`（動画＋gaze同期→距離→CSV）、
`code3`（CSV→統計→図→レポート）はいずれも合成データでエンドツーエンド実行を確認済み。
実データでは上記「設定の要点」を埋めてから回すこと。
