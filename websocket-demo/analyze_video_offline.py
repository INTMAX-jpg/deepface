"""Offline video analysis with bidirectional Gaussian smoothing.

Usage:
    python websocket-demo/analyze_video_offline.py --input demo.mp4 --output demo_result.json

Unlike the webcam pipeline, offline mode can see both the past and future around
frame t, so it smooths with t-K ... t+K and cuts smoothing at detected segment
boundaries.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from deepface import DeepFace

from emotion_enhance import (
    EMOTION_ORDER,
    QualityEstimator,
    bbox_iou,
    bidirectional_gaussian_smooth,
    crop_bbox,
    dominant_from_probs,
    frame_delta,
    frame_signature,
    normalize_emotion_scores,
    probs_to_percent_dict,
)


def normalize_face_area(area: Dict[str, Any]) -> Dict[str, int]:
    return {
        "x": int(area.get("x", 0)),
        "y": int(area.get("y", 0)),
        "w": int(area.get("w", 0)),
        "h": int(area.get("h", 0)),
    }


def convert_face_image(face_image: Any) -> Optional[np.ndarray]:
    if face_image is None:
        return None
    arr = np.asarray(face_image)
    if arr.size == 0:
        return None
    if arr.dtype != np.uint8:
        scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def analyze_largest_face(frame: np.ndarray, detector_backend: str = "retinaface", align: bool = True) -> Optional[Dict[str, Any]]:
    faces = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=align,
    )
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda item: float((item.get("facial_area") or {}).get("w", 0))
        * float((item.get("facial_area") or {}).get("h", 0)),
        reverse=True,
    )
    extracted = faces[0]
    face_image = convert_face_image(extracted.get("face"))
    if face_image is None:
        return None
    analysis_result = DeepFace.analyze(
        img_path=face_image,
        actions=["emotion"],
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )
    analysis = analysis_result[0] if isinstance(analysis_result, list) else analysis_result
    area = normalize_face_area(extracted.get("facial_area") or analysis.get("region") or {})
    face_conf = float(extracted.get("confidence", analysis.get("face_confidence", 0.0)) or 0.0)
    frame_h, frame_w = frame.shape[:2]
    area_ratio = (max(0, area.get("w", 0)) * max(0, area.get("h", 0))) / max(1, frame_w * frame_h)
    if area_ratio > 0.82 and face_conf <= 0.05:
        return None
    if area_ratio < 0.002:
        return None
    probs = normalize_emotion_scores(analysis.get("emotion", {}) or {})
    raw_emotion, raw_conf = dominant_from_probs(probs)
    return {
        "area": area,
        "face_conf": face_conf,
        "emotion_probs": probs_to_percent_dict(probs),
        "raw_emotion": raw_emotion,
        "raw_conf": round(raw_conf * 100.0, 1),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    quality_estimator = QualityEstimator()
    prev_sig = None
    prev_area = None
    no_face_streak = 0
    segment_id = 0
    frame_index = 0
    analyzed_attempts = 0
    no_face_count = 0
    discontinuity_count = 0
    analyzed_records: List[Dict[str, Any]] = []
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.every != 0:
            continue
        analyzed_attempts += 1
        timestamp = frame_index / fps
        delta = frame_delta(prev_sig, frame)
        if delta > args.frame_diff_threshold:
            segment_id += 1
            discontinuity_count += 1
            prev_area = None
            no_face_streak = 0
        face = analyze_largest_face(frame, detector_backend=args.detector_backend, align=not args.no_align)
        if face is None:
            no_face_count += 1
            no_face_streak += 1
            if no_face_streak >= args.no_face_gap:
                segment_id += 1
                discontinuity_count += 1
                no_face_streak = 0
                prev_area = None
            prev_sig = frame_signature(frame)
            continue
        no_face_streak = 0
        quality, details = quality_estimator.estimate(frame, face["area"], face.get("face_conf", 0.0), prev_area)
        if prev_area is not None and bbox_iou(prev_area, face["area"]) < 0.08 and delta > args.frame_diff_threshold * 0.6:
            segment_id += 1
            discontinuity_count += 1
        record = {
            "frame_index": frame_index,
            "timestamp": round(timestamp, 3),
            "segment_id": segment_id,
            "area": face["area"],
            "face_conf": round(float(face.get("face_conf", 0.0)), 4),
            "quality_score": quality,
            "quality_details": details,
            "emotion_probs": face["emotion_probs"],
            "raw_emotion": face["raw_emotion"],
            "raw_conf": face["raw_conf"],
            "frame_delta": round(delta, 4),
        }
        analyzed_records.append(record)
        prev_area = face["area"]
        prev_sig = frame_signature(frame)

    cap.release()
    smoothed = bidirectional_gaussian_smooth(analyzed_records, window=args.window, sigma=args.sigma)
    elapsed = time.perf_counter() - start
    return {
        "input": str(args.input),
        "fps": fps,
        "every": args.every,
        "window": args.window,
        "sigma": args.sigma,
        "detector_backend": args.detector_backend,
        "emotion_order": EMOTION_ORDER,
        "total_frames_read": frame_index,
        "analyzed_attempts": analyzed_attempts,
        "no_face_count": no_face_count,
        "no_face_rate": round(no_face_count / analyzed_attempts, 4) if analyzed_attempts else None,
        "segment_count": len({record.get("segment_id", 0) for record in smoothed}),
        "discontinuity_count": discontinuity_count,
        "num_analyzed_faces": len(smoothed),
        "elapsed_seconds": round(elapsed, 2),
        "records": smoothed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--every", type=int, default=5, help="Analyze one frame every N video frames")
    parser.add_argument("--detector_backend", default="retinaface")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--no-face-gap", dest="no_face_gap", type=int, default=5)
    parser.add_argument("--frame-diff-threshold", dest="frame_diff_threshold", type=float, default=0.48)
    parser.add_argument("--no-align", action="store_true")
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(result['records'])} analyzed records")


if __name__ == "__main__":
    main()
