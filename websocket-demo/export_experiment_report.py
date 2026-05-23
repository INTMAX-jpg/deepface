"""Export report-ready metrics and plots from offline video analysis JSON.

Typical workflow:
    python analyze_video_offline.py --input demo.mp4 --output outputs/demo_result.json
    python export_experiment_report.py --input outputs/demo_result.json --output-dir outputs/report_demo

Outputs:
    summary.json        machine-readable metrics
    summary.csv         one-row table for the report
    frame_records.csv   per-frame raw/smooth/quality records
    emotion_curve.png   raw vs smooth dominant-emotion curve, if matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

EMOTION_ORDER = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTION_ORDER)}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def switch_count(records: List[Dict[str, Any]], key: str) -> int:
    """Count emotion changes without crossing segment boundaries."""
    count = 0
    prev_emotion: Optional[str] = None
    prev_segment: Optional[int] = None
    for rec in records:
        emotion = rec.get(key)
        segment = int(rec.get("segment_id", 0))
        if not emotion:
            continue
        if prev_emotion is not None and prev_segment == segment and emotion != prev_emotion:
            count += 1
        prev_emotion = emotion
        prev_segment = segment
    return count


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    records = payload.get("records", []) or []
    raw_switches = switch_count(records, "raw_emotion")
    smooth_switches = switch_count(records, "smooth_emotion")
    reduction = None
    if raw_switches > 0:
        reduction = round((raw_switches - smooth_switches) / raw_switches, 4)
    segments = sorted({int(rec.get("segment_id", 0)) for rec in records})
    avg_quality = safe_mean([float(rec.get("quality_score", 0.0)) for rec in records])
    analyzed_attempts = payload.get("analyzed_attempts")
    no_face_count = payload.get("no_face_count")
    no_face_rate = payload.get("no_face_rate")
    if no_face_rate is None and analyzed_attempts:
        no_face_rate = round(float(no_face_count or 0) / float(analyzed_attempts), 4)

    return {
        "input": payload.get("input"),
        "fps": payload.get("fps"),
        "analyze_every_n_frames": payload.get("every"),
        "smoothing_window": payload.get("window"),
        "smoothing_sigma": payload.get("sigma"),
        "detector_backend": payload.get("detector_backend"),
        "total_frames_read": payload.get("total_frames_read"),
        "analyzed_attempts": analyzed_attempts,
        "num_analyzed_faces": payload.get("num_analyzed_faces", len(records)),
        "no_face_count": no_face_count,
        "no_face_rate": no_face_rate,
        "average_quality_score": round(avg_quality, 4) if avg_quality is not None else None,
        "segment_count": payload.get("segment_count", len(segments)),
        "discontinuity_count": payload.get("discontinuity_count"),
        "raw_emotion_switch_count": raw_switches,
        "smooth_emotion_switch_count": smooth_switches,
        "switch_reduction_ratio": reduction,
        "switch_reduction_percent": round(reduction * 100.0, 2) if reduction is not None else None,
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def write_records_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "timestamp",
        "segment_id",
        "raw_emotion",
        "raw_conf",
        "smooth_emotion",
        "smooth_conf",
        "quality_score",
        "face_conf",
        "frame_delta",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key) for key in fieldnames})


def plot_curve(path: Path, records: List[Dict[str, Any]]) -> bool:
    if not records:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = [int(rec.get("frame_index", idx)) for idx, rec in enumerate(records)]
    raw_y = [EMOTION_TO_ID.get(str(rec.get("raw_emotion")), -1) for rec in records]
    smooth_y = [EMOTION_TO_ID.get(str(rec.get("smooth_emotion")), -1) for rec in records]
    quality = [float(rec.get("quality_score", 0.0)) * (len(EMOTION_ORDER) - 1) for rec in records]

    plt.figure(figsize=(12, 5))
    plt.plot(xs, raw_y, marker="o", linewidth=1, markersize=3, label="raw dominant emotion")
    plt.plot(xs, smooth_y, marker="o", linewidth=2, markersize=3, label="smoothed dominant emotion")
    plt.plot(xs, quality, linestyle="--", linewidth=1, label="quality score (scaled)")
    plt.yticks(list(range(len(EMOTION_ORDER))), EMOTION_ORDER)
    plt.xlabel("frame index")
    plt.ylabel("emotion")
    plt.title("Raw vs Gaussian-smoothed dominant emotion")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="JSON generated by analyze_video_offline.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = load_json(args.input)
    records = payload.get("records", []) or []
    summary = summarize(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(args.output_dir / "summary.csv", summary)
    write_records_csv(args.output_dir / "frame_records.csv", records)
    plotted = plot_curve(args.output_dir / "emotion_curve.png", records)

    print("Experiment report exported:")
    print(f"  summary.json:      {args.output_dir / 'summary.json'}")
    print(f"  summary.csv:       {args.output_dir / 'summary.csv'}")
    print(f"  frame_records.csv: {args.output_dir / 'frame_records.csv'}")
    if plotted:
        print(f"  emotion_curve.png: {args.output_dir / 'emotion_curve.png'}")
    else:
        print("  emotion_curve.png: skipped (no records or matplotlib unavailable)")


if __name__ == "__main__":
    main()
