import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Prediction:
    image: str
    method: str
    angle: Optional[float]
    true_angle: Optional[float]
    error: Optional[float]
    status: str


def parse_label_from_filename(path: Path) -> Optional[float]:
    """Return the angle encoded after the last underscore, or None for unlabeled files."""
    stem = path.stem
    if "_" not in stem:
        return None
    token = stem.rsplit("_", 1)[-1]
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    return None


def iter_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


def load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install it with: pip install opencv-python"
        ) from exc
    return cv2


def imread_unicode(path: Path):
    cv2 = load_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    cv2 = load_cv2()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def minimal_circular_span(angles_deg: np.ndarray, trim_percent: float = 0.5) -> float:
    """Smallest arc that contains the angular samples after light outlier trimming."""
    if angles_deg.size < 20:
        return float("nan")
    angles = np.sort(np.mod(angles_deg, 360.0))
    if trim_percent > 0:
        lo, hi = np.percentile(angles, [trim_percent, 100.0 - trim_percent])
        trimmed = angles[(angles >= lo) & (angles <= hi)]
        if trimmed.size >= 20:
            angles = trimmed
    diffs = np.diff(np.r_[angles, angles[0] + 360.0])
    return float(360.0 - np.max(diffs))


def get_preprocessor(top_model: Path, side_model: Path):
    try:
        from preprocess_interface import ValveAIPreprocessor
    except Exception as exc:
        raise RuntimeError(f"Cannot import preprocess_interface.py: {exc}") from exc
    return ValveAIPreprocessor(top_model=str(top_model), side_model=str(side_model))


def maybe_preprocess(path: Path, preprocessor, enabled: bool):
    if not enabled:
        img = imread_unicode(path)
        if img is None:
            return None, None
        h, w = img.shape[:2]
        return img, (0.5, 0.5, 0.5, 0.5)
    roi, coords = preprocessor.process(str(path), view_type="top")
    return roi, coords


def opencv_angle_from_image(
    image: np.ndarray,
    center_xy: tuple[float, float],
    green_hsv: tuple[int, int, int, int, int, int],
    red_hsv1: tuple[int, int, int, int, int, int],
    red_hsv2: tuple[int, int, int, int, int, int],
    radius_quantiles: tuple[float, float],
    mode: str,
    total_window_angle: float,
    span_trim_percent: float,
    hybrid_span_weight: float,
    hybrid_diff_threshold: float,
    hybrid_fallback_span_weight: float,
) -> float:
    cv2 = load_cv2()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_low = np.array(green_hsv[:3], dtype=np.uint8)
    green_high = np.array(green_hsv[3:], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, green_low, green_high)
    green_mask = clean_mask(green_mask)

    if mode in {"area", "hybrid"}:
        red1 = cv2.inRange(
            hsv,
            np.array(red_hsv1[:3], dtype=np.uint8),
            np.array(red_hsv1[3:], dtype=np.uint8),
        )
        red2 = cv2.inRange(
            hsv,
            np.array(red_hsv2[:3], dtype=np.uint8),
            np.array(red_hsv2[3:], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(red1, red2)
        red_mask = clean_mask(red_mask)
        green_area = int(np.count_nonzero(green_mask))
        red_area = int(np.count_nonzero(red_mask))
        total_area = green_area + red_area
        if total_area == 0:
            area_angle = float("nan")
        else:
            area_angle = float(total_window_angle * green_area / total_area)
        if mode == "area":
            return area_angle

    ys, xs = np.where(green_mask > 0)
    if xs.size < 30:
        return area_angle if mode == "hybrid" else float("nan")

    cx, cy = center_xy
    radii = np.hypot(xs - cx, ys - cy)
    r_lo, r_hi = np.percentile(radii, radius_quantiles)
    annulus = (radii >= r_lo) & (radii <= r_hi)
    if np.count_nonzero(annulus) >= 30:
        xs = xs[annulus]
        ys = ys[annulus]

    angles = np.degrees(np.arctan2(ys - cy, xs - cx))
    span_angle = minimal_circular_span(angles, span_trim_percent)
    if mode == "hybrid" and math.isfinite(area_angle):
        if not math.isfinite(span_angle):
            return area_angle
        if area_angle - span_angle > hybrid_diff_threshold:
            weight = hybrid_fallback_span_weight
        else:
            weight = hybrid_span_weight
        weight = min(1.0, max(0.0, weight))
        return float(weight * span_angle + (1.0 - weight) * area_angle)
    return span_angle


def fit_linear_calibration(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    pairs = [(p, t) for p, t in pairs if math.isfinite(p) and math.isfinite(t)]
    if len(pairs) < 2:
        return 1.0, 0.0
    pred = np.array([p for p, _ in pairs], dtype=np.float64)
    true = np.array([t for _, t in pairs], dtype=np.float64)
    a, b = np.polyfit(pred, true, 1)
    return float(a), float(b)


def build_resnet50_regression(device):
    import torch.nn as nn
    from torchvision import models

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return model.to(device)


class AIAnglePredictor:
    def __init__(self, model_path: Path):
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and torchvision are required. Install them with: "
                "pip install torch torchvision"
            ) from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model = build_resnet50_regression(self.device)
        state = torch.load(str(model_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, bgr_image: np.ndarray) -> float:
        cv2 = load_cv2()
        from PIL import Image

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return float(self.model(tensor).item())


def summarize(predictions: list[Prediction], method: str) -> str:
    rows = [p for p in predictions if p.method == method and p.error is not None]
    if not rows:
        return f"{method}: no labeled images for metrics"
    errors = np.array([p.error for p in rows], dtype=np.float64)
    return (
        f"{method}: n={len(rows)}, MAE={errors.mean():.3f}, "
        f"MaxAE={errors.max():.3f}"
    )


def save_csv(path: Path, rows: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "method", "angle", "true_angle", "error", "status"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row.image,
                    "method": row.method,
                    "angle": "" if row.angle is None else f"{row.angle:.4f}",
                    "true_angle": ""
                    if row.true_angle is None
                    else f"{row.true_angle:.4f}",
                    "error": "" if row.error is None else f"{row.error:.4f}",
                    "status": row.status,
                }
            )


def run(args) -> list[Prediction]:
    root = Path(args.input).resolve()
    images = iter_images(root)
    if not images:
        raise RuntimeError(f"No images found under {root}")

    use_preprocess = args.preprocess != "none"
    preprocessor = None
    if use_preprocess:
        preprocessor = get_preprocessor(Path(args.top_model), Path(args.side_model))

    calibration = {"opencv": (1.0, 0.0), "ai": (1.0, 0.0)}
    green_hsv = tuple(args.green_hsv)
    red_hsv1 = tuple(args.red_hsv1)
    red_hsv2 = tuple(args.red_hsv2)
    radius_quantiles = tuple(args.radius_quantiles)

    ai_predictor = None
    if args.method in {"ai", "both"}:
        ai_predictor = AIAnglePredictor(Path(args.angle_model))

    if args.calibrate_dir:
        cal_images = iter_images(Path(args.calibrate_dir).resolve())
        for method in ["opencv", "ai"]:
            if args.method not in {method, "both"}:
                continue
            if method == "opencv" and args.opencv_calibration == "none":
                print("opencv calibration: disabled")
                continue
            pairs: list[tuple[float, float]] = []
            for path in cal_images:
                true = parse_label_from_filename(path)
                if true is None:
                    continue
                image, coords = maybe_preprocess(path, preprocessor, use_preprocess)
                if image is None or coords is None:
                    continue
                h, w = image.shape[:2]
                cx, cy = float(coords[0]) * w, float(coords[1]) * h
                if method == "opencv":
                    pred = opencv_angle_from_image(
                        image,
                        (cx, cy),
                        green_hsv,
                        red_hsv1,
                        red_hsv2,
                        radius_quantiles,
                        args.opencv_mode,
                        args.total_window_angle,
                        args.opencv_span_trim,
                        args.opencv_hybrid_span_weight,
                        args.opencv_hybrid_diff_threshold,
                        args.opencv_hybrid_fallback_span_weight,
                    )
                else:
                    pred = ai_predictor.predict(image)
                pairs.append((pred, true))
            calibration[method] = fit_linear_calibration(pairs)
            a, b = calibration[method]
            print(f"{method} calibration: angle = {a:.6f} * raw + {b:.6f}")

    results: list[Prediction] = []
    for path in images:
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        true = parse_label_from_filename(path)
        image, coords = maybe_preprocess(path, preprocessor, use_preprocess)
        if image is None or coords is None:
            for method in ["opencv", "ai"]:
                if args.method in {method, "both"}:
                    results.append(Prediction(rel, method, None, true, None, "no_roi"))
            continue

        h, w = image.shape[:2]
        cx, cy = float(coords[0]) * w, float(coords[1]) * h

        if args.method in {"opencv", "both"}:
            raw = opencv_angle_from_image(
                image,
                (cx, cy),
                green_hsv,
                red_hsv1,
                red_hsv2,
                radius_quantiles,
                args.opencv_mode,
                args.total_window_angle,
                args.opencv_span_trim,
                args.opencv_hybrid_span_weight,
                args.opencv_hybrid_diff_threshold,
                args.opencv_hybrid_fallback_span_weight,
            )
            a, b = calibration["opencv"]
            angle = raw * a + b if math.isfinite(raw) else None
            if angle is not None:
                angle = angle * args.opencv_output_scale + args.opencv_output_offset
            error = None if true is None or angle is None else abs(angle - true)
            results.append(
                Prediction(rel, "opencv", angle, true, error, "ok" if angle is not None else "no_green")
            )

        if args.method in {"ai", "both"}:
            raw = ai_predictor.predict(image)
            a, b = calibration["ai"]
            angle = raw * a + b
            error = None if true is None else abs(angle - true)
            results.append(Prediction(rel, "ai", angle, true, error, "ok"))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the central angle of the top green valve sector."
    )
    parser.add_argument("--input", default="data_samples/top_test")
    parser.add_argument("--method", choices=["opencv", "ai", "both"], default="both")
    parser.add_argument("--output", default="outputs/top_angle_predictions.csv")
    parser.add_argument("--calibrate-dir", default="data_samples/top")
    parser.add_argument("--preprocess", choices=["yolo", "none"], default="yolo")
    parser.add_argument("--top-model", default="top_best.pt")
    parser.add_argument("--side-model", default="side_best.pt")
    parser.add_argument(
        "--angle-model",
        default="outputs/top_angle_resnet50_retrained_300_best.pth",
    )
    parser.add_argument(
        "--green-hsv",
        nargs=6,
        type=int,
        default=[40, 80, 80, 100, 255, 255],
        metavar=("H1", "S1", "V1", "H2", "S2", "V2"),
    )
    parser.add_argument(
        "--red-hsv1",
        nargs=6,
        type=int,
        default=[0, 124, 124, 10, 255, 255],
        metavar=("H1", "S1", "V1", "H2", "S2", "V2"),
    )
    parser.add_argument(
        "--red-hsv2",
        nargs=6,
        type=int,
        default=[160, 124, 124, 180, 255, 255],
        metavar=("H1", "S1", "V1", "H2", "S2", "V2"),
    )
    parser.add_argument("--opencv-mode", choices=["area", "span", "hybrid"], default="hybrid")
    parser.add_argument(
        "--opencv-calibration",
        choices=["none", "linear"],
        default="none",
        help="Apply linear calibration to OpenCV predictions. AI calibration is unchanged.",
    )
    parser.add_argument("--total-window-angle", type=float, default=80.0)
    parser.add_argument(
        "--radius-quantiles",
        nargs=2,
        type=float,
        default=[60.0, 95.0],
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--opencv-span-trim",
        type=float,
        default=2.0,
        help="Percent of angular outliers to trim at each end for OpenCV span mode.",
    )
    parser.add_argument(
        "--opencv-hybrid-span-weight",
        type=float,
        default=0.24,
        help="Blend weight for the span estimate in OpenCV hybrid mode.",
    )
    parser.add_argument(
        "--opencv-hybrid-diff-threshold",
        type=float,
        default=10.0,
        help="If area estimate exceeds span estimate by this many degrees, use fallback span weight.",
    )
    parser.add_argument(
        "--opencv-hybrid-fallback-span-weight",
        type=float,
        default=1.0,
        help="Hybrid span weight used when area and span estimates strongly disagree.",
    )
    parser.add_argument("--opencv-output-scale", type=float, default=0.9625)
    parser.add_argument("--opencv-output-offset", type=float, default=0.746)
    args = parser.parse_args()

    rows = run(args)
    save_csv(Path(args.output), rows)

    print(f"saved: {Path(args.output).resolve()}")
    for method in ["opencv", "ai"]:
        if args.method in {method, "both"}:
            print(summarize(rows, method))


if __name__ == "__main__":
    main()
