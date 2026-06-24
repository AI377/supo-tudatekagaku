"""
geom_utils.py
3コード共通のユーティリティ。
- 設定読み込み
- 座標/距離変換（ピクセル ⇔ 正規化 ⇔ 視野角[度]）
- ボール検出器（YOLO + 色フォールバック）
- 日本語フォント設定
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pyyaml 未導入でも import 自体は通す
    yaml = None


# ------------------------------------------------------------
# 設定読み込み
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    """YAML 設定ファイルを読み込む。"""
    if yaml is None:
        raise ImportError("pyyaml が必要です: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# 座標・距離の変換
# ------------------------------------------------------------
def bbox_center(box_xyxy) -> Tuple[float, float]:
    """[x1, y1, x2, y2] -> 中心 (cx, cy)"""
    x1, y1, x2, y2 = box_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """2点間のユークリッド距離[px]"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def frame_center(width: int, height: int) -> Tuple[float, float]:
    """フレーム中心（=頭部正面方向の代理）"""
    return (width / 2.0, height / 2.0)


def diagonal(width: int, height: int) -> float:
    return math.hypot(width, height)


def normalize_by_diagonal(dist_px: float, width: int, height: int) -> float:
    """対角長で正規化（0-1付近）。デバイス間の解像度差を吸収する簡易指標。"""
    d = diagonal(width, height)
    return dist_px / d if d > 0 else float("nan")


def focal_px_from_fov(width: int, fov_h_deg: float) -> float:
    """水平FOV[度]と幅Wからピンホール近似の焦点距離[px]を求める。"""
    fov_rad = math.radians(fov_h_deg)
    return (width / 2.0) / math.tan(fov_rad / 2.0)


def px_to_deg(dist_px: float, width: int, fov_h_deg: float) -> float:
    """
    画面中心からのピクセル距離を視野角[度]へ換算（ピンホール近似）。
    angle = atan(r / f) ,  f = (W/2)/tan(FOV/2)
    ※ 魚眼の強い歪みがある場合は事前に undistort してから使うこと。
    """
    if fov_h_deg is None or width <= 0:
        return float("nan")
    f = focal_px_from_fov(width, fov_h_deg)
    return math.degrees(math.atan2(dist_px, f))


# ------------------------------------------------------------
# 魚眼歪み補正（任意）
# ------------------------------------------------------------
def build_fisheye_maps(K, D, size: Tuple[int, int]):
    """GoPro魚眼の undistort 用マップを作る。K,D はキャリブレーション値。"""
    import cv2
    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64)
    w, h = size
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2


# ------------------------------------------------------------
# ボール検出器
# ------------------------------------------------------------
@dataclass
class Detection:
    """1フレームのボール検出結果。"""
    found: bool
    cx: float = float("nan")
    cy: float = float("nan")
    conf: float = float("nan")
    box: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2


class BallDetector:
    """
    YOLO（ultralytics）を優先し、使えない場合は HSV 色検出にフォールバック。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.det_cfg = cfg["detect"]
        self.color_cfg = cfg.get("color_fallback", {"enabled": False})
        self.model = None
        self.ball_class_id = self.det_cfg.get("ball_class_id")
        self.mode = "color"
        self._init_yolo()

    def _init_yolo(self):
        model_path = self.det_cfg.get("model_path")
        if not model_path or not os.path.exists(model_path):
            print(f"[BallDetector] YOLOモデルが見つかりません({model_path}) -> 色検出にフォールバック")
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[BallDetector] ultralytics 未導入 -> 色検出にフォールバック")
            return
        self.model = YOLO(model_path)
        self.mode = "yolo"
        # クラスID自動解決（"ball" という名前を探す）
        if self.ball_class_id is None and hasattr(self.model, "names"):
            for cid, name in self.model.names.items():
                if str(name).lower() == "ball":
                    self.ball_class_id = int(cid)
                    break
        print(f"[BallDetector] YOLO使用: {model_path} (ball_class_id={self.ball_class_id})")

    # -- YOLO 検出 --
    def _detect_yolo(self, frame) -> list:
        device = self.det_cfg.get("device", "auto")
        kwargs = dict(
            conf=self.det_cfg.get("conf_threshold", 0.25),
            imgsz=self.det_cfg.get("imgsz", 640),
            verbose=False,
        )
        if device and device != "auto":
            kwargs["device"] = device
        results = self.model.predict(frame, **kwargs)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                if self.ball_class_id is not None and cls_id != self.ball_class_id:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                conf = float(b.conf[0])
                cx, cy = bbox_center((x1, y1, x2, y2))
                dets.append(Detection(True, cx, cy, conf, (x1, y1, x2, y2)))
        return dets

    # -- 色検出（フォールバック）--
    def _detect_color(self, frame) -> list:
        if not self.color_cfg.get("enabled", False):
            return []
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array(self.color_cfg["hsv_lower"], dtype=np.uint8)
        hi = np.array(self.color_cfg["hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        min_area = self.color_cfg.get("min_area", 80)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2.0, y + h / 2.0
            # 面積を擬似的な信頼度に使う（円形度で重み付けしてもよい）
            dets.append(Detection(True, cx, cy, float(area), (x, y, x + w, y + h)))
        return dets

    def detect(self, frame) -> list:
        if self.mode == "yolo":
            dets = self._detect_yolo(frame)
            if dets:
                return dets
            # YOLOで取れないフレームは色検出で補完
            return self._detect_color(frame)
        return self._detect_color(frame)


def select_ball(dets: list, prev_xy, width: int, height: int, cfg: dict) -> Detection:
    """
    複数検出から1個を選ぶ。
    - 前フレーム位置があり tracking 有効なら最近傍を優先（大きく飛ぶものは除外）
    - それ以外は信頼度（or 面積）最大
    """
    if not dets:
        return Detection(False)

    trk = cfg.get("tracking", {})
    if trk.get("use_prev_nearest", False) and prev_xy is not None and not _is_nan(prev_xy):
        diag = diagonal(width, height)
        max_jump = trk.get("max_jump_norm", 0.3) * diag
        candidates = [
            (euclidean((d.cx, d.cy), prev_xy), d)
            for d in dets
            if euclidean((d.cx, d.cy), prev_xy) <= max_jump
        ]
        if candidates:
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]
    # フォールバック：信頼度最大
    return max(dets, key=lambda d: d.conf)


def _is_nan(xy) -> bool:
    return xy is None or (isinstance(xy[0], float) and math.isnan(xy[0]))


# ------------------------------------------------------------
# 日本語フォント設定（matplotlib）
# ------------------------------------------------------------
def setup_japanese_font():
    """matplotlib で日本語が文字化けしないように設定する。"""
    import matplotlib
    try:
        import japanize_matplotlib  # noqa: F401  import するだけで適用される
        return
    except ImportError:
        pass
    # フォールバック：システムにありそうな日本語フォントを順に試す
    candidates = [
        "Noto Sans CJK JP", "IPAexGothic", "IPAGothic",
        "TakaoGothic", "VL Gothic", "Hiragino Sans", "Yu Gothic", "MS Gothic",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
