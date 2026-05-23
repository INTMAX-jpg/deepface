import argparse
import json
import statistics
import time
from typing import Any, Callable, Dict, List

import cv2

from deepface import DeepFace
from deepface.modules import streaming


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def summarize_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "avg_ms": statistics.mean(values) * 1000,
        "p95_ms": percentile(values, 0.95) * 1000,
    }


def timed_call(samples: List[float], fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            samples.append(time.perf_counter() - start)

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["old", "new"], required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--source", default="0")
    parser.add_argument("--db-path", default="face_db")
    parser.add_argument("--detector-backend", default="opencv")
    parser.add_argument("--process-width", type=int, default=640)
    parser.add_argument("--emotion-interval", type=int, default=3)
    args = parser.parse_args()

    source: Any = int(args.source) if args.source.isdigit() else args.source
    start_time = time.perf_counter()
    imshow_times: List[float] = []
    detect_times: List[float] = []
    analyze_times: List[float] = []

    original_imshow = cv2.imshow
    original_wait_key = cv2.waitKey
    original_grab = streaming.grab_facial_areas
    original_analyze = DeepFace.analyze

    def benchmark_imshow(*_args: Any, **_kwargs: Any) -> None:
        imshow_times.append(time.perf_counter())

    def benchmark_wait_key(delay: int = 1) -> int:
        if time.perf_counter() - start_time >= args.duration:
            return ord("q")
        return -1

    cv2.imshow = benchmark_imshow
    cv2.waitKey = benchmark_wait_key
    streaming.grab_facial_areas = timed_call(detect_times, original_grab)
    DeepFace.analyze = timed_call(analyze_times, original_analyze)

    try:
        if args.mode == "old":
            DeepFace.stream(
                db_path=args.db_path,
                source=source,
                detector_backend=args.detector_backend,
                enable_face_analysis=True,
            )
        else:
            DeepFace.stream_emotion(
                source=source,
                detector_backend=args.detector_backend,
                process_width=args.process_width,
                emotion_interval=args.emotion_interval,
            )
    finally:
        cv2.imshow = original_imshow
        cv2.waitKey = original_wait_key
        streaming.grab_facial_areas = original_grab
        DeepFace.analyze = original_analyze

    elapsed = max(time.perf_counter() - start_time, 0.001)
    frame_intervals = [
        imshow_times[index] - imshow_times[index - 1] for index in range(1, len(imshow_times))
    ]
    frame_times_ms = [value * 1000 for value in frame_intervals]

    result = {
        "mode": args.mode,
        "elapsed_sec": elapsed,
        "display_frames": len(imshow_times),
        "display_fps": len(imshow_times) / elapsed,
        "emotion_fps": len(analyze_times) / elapsed,
        "frame_time_avg_ms": statistics.mean(frame_times_ms) if frame_times_ms else 0.0,
        "frame_time_p95_ms": percentile(frame_times_ms, 0.95),
        "detect": summarize_ms(detect_times),
        "analyze": summarize_ms(analyze_times),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
