"""Enhanced temporal smoothing and personalized expression adaptation for the demo.

This module is intentionally lightweight: it only depends on numpy / cv2 and the
DeepFace package that already ships with this repository.  It adds two project
features without forcing the UI to depend on a new heavy training framework:

1. Gaussian temporal smoothing with frame-quality penalties and segment cuts.
2. Local face profiles plus a tiny per-person softmax adapter trained from
   manually-labelled DeepFace emotion probability vectors.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import re
import time
from collections import deque
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

EMOTION_ORDER = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
NEGATIVE_EMOTIONS = {"angry", "fear", "sad", "disgust"}


def now_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def slugify_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", name.strip())
    cleaned = cleaned.strip("_") or "person"
    return cleaned[:40]


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a browser data URL/base64 string into a BGR OpenCV image."""
    if image_base64.startswith("data:image/"):
        image_base64 = image_base64.split(",", 1)[1]
    img_data = base64.b64decode(image_base64)
    arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解析图像，请确认上传的是浏览器截图或普通图片")
    return img


def normalize_emotion_scores(scores: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return a probability distribution over EMOTION_ORDER.

    DeepFace emotion dictionaries are usually percentages.  This function accepts
    percentages or probabilities, clamps negatives, fills missing emotions, and
    normalizes to sum to 1.0.
    """
    scores = scores or {}
    values = np.array([max(0.0, float(scores.get(e, 0.0))) for e in EMOTION_ORDER], dtype=np.float64)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 1e-12:
        values = np.ones(len(EMOTION_ORDER), dtype=np.float64) / len(EMOTION_ORDER)
    else:
        values = values / total
    return {emotion: float(values[i]) for i, emotion in enumerate(EMOTION_ORDER)}


def probs_to_percent_dict(probs: Dict[str, float], decimals: int = 1) -> Dict[str, float]:
    return {emotion: round(float(probs.get(emotion, 0.0)) * 100.0, decimals) for emotion in EMOTION_ORDER}


def dominant_from_probs(probs: Dict[str, float]) -> Tuple[str, float]:
    emotion = max(EMOTION_ORDER, key=lambda key: float(probs.get(key, 0.0)))
    return emotion, float(probs.get(emotion, 0.0))


def emotion_vector(probs: Dict[str, float], quality_score: float = 1.0) -> np.ndarray:
    return np.array([float(probs.get(e, 0.0)) for e in EMOTION_ORDER] + [float(quality_score)], dtype=np.float64)


def bbox_iou(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
    ax2, ay2 = ax1 + max(0.0, float(a.get("w", 0))), ay1 + max(0.0, float(a.get("h", 0)))
    bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
    bx2, by2 = bx1 + max(0.0, float(b.get("w", 0))), by1 + max(0.0, float(b.get("h", 0)))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def crop_bbox(frame: np.ndarray, area: Dict[str, Any], expand: float = 0.12) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x, y = int(area.get("x", 0)), int(area.get("y", 0))
    bw, bh = int(area.get("w", 0)), int(area.get("h", 0))
    if bw <= 2 or bh <= 2:
        return None
    pad_x, pad_y = int(bw * expand), int(bh * expand)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def frame_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)


def frame_delta(prev_sig: Optional[np.ndarray], curr_frame: np.ndarray) -> float:
    if prev_sig is None:
        return 0.0
    curr_sig = frame_signature(curr_frame)
    return float(np.mean(np.abs(curr_sig.astype(np.float32) - prev_sig.astype(np.float32))) / 255.0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


class QualityEstimator:
    """Quality score for bad/ambiguous frames.

    The score is deliberately explainable for reports:
    face confidence + size + blur + center + bbox stability.
    """

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def estimate(
        self,
        frame: np.ndarray,
        area: Dict[str, Any],
        face_confidence: float,
        previous_area: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        height, width = frame.shape[:2]
        x, y = float(area.get("x", 0)), float(area.get("y", 0))
        bw, bh = max(0.0, float(area.get("w", 0))), max(0.0, float(area.get("h", 0)))
        frame_area = max(1.0, float(width * height))
        face_ratio = (bw * bh) / frame_area

        q_face = self._clip01(face_confidence if face_confidence > 0 else 0.45)
        q_area = self._clip01((face_ratio - 0.008) / 0.09)

        crop = crop_bbox(frame, area, expand=0.02)
        if crop is None or crop.size == 0:
            q_blur = 0.0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            q_blur = self._clip01((lap_var - 12.0) / 160.0)

        center_x, center_y = x + bw / 2.0, y + bh / 2.0
        norm_dx = abs(center_x - width / 2.0) / max(1.0, width / 2.0)
        norm_dy = abs(center_y - height / 2.0) / max(1.0, height / 2.0)
        q_center = self._clip01(1.0 - 0.55 * norm_dx - 0.45 * norm_dy)

        if previous_area is None:
            q_bbox_stability = 1.0
        else:
            q_bbox_stability = self._clip01((bbox_iou(previous_area, area) - 0.05) / 0.55)

        quality = (
            0.35 * q_face
            + 0.20 * q_area
            + 0.20 * q_blur
            + 0.15 * q_center
            + 0.10 * q_bbox_stability
        )
        details = {
            "face": round(q_face, 3),
            "area": round(q_area, 3),
            "blur": round(q_blur, 3),
            "center": round(q_center, 3),
            "bbox_stability": round(q_bbox_stability, 3),
        }
        return round(self._clip01(quality), 3), details


@dataclass
class SmoothRecord:
    frame_index: int
    segment_id: int
    probs: Dict[str, float]
    quality_score: float
    timestamp: float


class CausalGaussianSmoother:
    """Streaming smoother: only past frames and current frame are visible."""

    def __init__(self, window: int = 5, sigma: float = 2.0):
        self.window = int(max(1, window))
        self.sigma = float(max(0.3, sigma))
        self.buffers: Dict[str, Deque[SmoothRecord]] = {}

    def reset(self) -> None:
        self.buffers.clear()

    def update(
        self,
        key: str,
        probs: Dict[str, float],
        quality_score: float,
        frame_index: int,
        segment_id: int,
        timestamp: Optional[float] = None,
        window: Optional[int] = None,
        sigma: Optional[float] = None,
    ) -> Dict[str, float]:
        window = int(max(1, window or self.window))
        sigma = float(max(0.3, sigma or self.sigma))
        buf = self.buffers.setdefault(key, deque(maxlen=max(2 * window + 1, window + 2)))
        buf.append(SmoothRecord(frame_index, segment_id, normalize_emotion_scores(probs), float(quality_score), timestamp or time.time()))

        weighted = np.zeros(len(EMOTION_ORDER), dtype=np.float64)
        total_weight = 0.0
        for record in buf:
            if record.segment_id != segment_id:
                continue
            dt = frame_index - record.frame_index
            if dt < 0 or dt > window:
                continue
            temporal_w = math.exp(-float(dt * dt) / (2.0 * sigma * sigma))
            quality_w = max(0.03, min(1.0, float(record.quality_score)))
            weight = temporal_w * quality_w
            weighted += weight * np.array([record.probs[e] for e in EMOTION_ORDER], dtype=np.float64)
            total_weight += weight

        if total_weight <= 1e-12:
            return normalize_emotion_scores(probs)
        weighted /= total_weight
        return {emotion: float(weighted[i]) for i, emotion in enumerate(EMOTION_ORDER)}


def bidirectional_gaussian_smooth(
    records: List[Dict[str, Any]],
    window: int = 5,
    sigma: float = 2.0,
) -> List[Dict[str, Any]]:
    """Offline video smoother: uses t-K ... t+K, but never crosses segment_id."""
    window = int(max(1, window))
    sigma = float(max(0.3, sigma))
    output: List[Dict[str, Any]] = []
    for t, target in enumerate(records):
        segment_id = target.get("segment_id", 0)
        weighted = np.zeros(len(EMOTION_ORDER), dtype=np.float64)
        total_weight = 0.0
        start = max(0, t - window)
        end = min(len(records), t + window + 1)
        for i in range(start, end):
            rec = records[i]
            if rec.get("segment_id", 0) != segment_id:
                continue
            dt = i - t
            temporal_w = math.exp(-float(dt * dt) / (2.0 * sigma * sigma))
            quality_w = max(0.03, min(1.0, float(rec.get("quality_score", 1.0))))
            weight = temporal_w * quality_w
            probs = normalize_emotion_scores(rec.get("emotion_probs", {}))
            weighted += weight * np.array([probs[e] for e in EMOTION_ORDER], dtype=np.float64)
            total_weight += weight
        if total_weight <= 1e-12:
            smoothed = normalize_emotion_scores(target.get("emotion_probs", {}))
        else:
            weighted /= total_weight
            smoothed = {emotion: float(weighted[j]) for j, emotion in enumerate(EMOTION_ORDER)}
        dominant, conf = dominant_from_probs(smoothed)
        merged = dict(target)
        merged["smooth_emotion_probs"] = probs_to_percent_dict(smoothed)
        merged["smooth_emotion"] = dominant
        merged["smooth_conf"] = round(conf * 100.0, 1)
        output.append(merged)
    return output


class SoftmaxEmotionAdapter:
    """Tiny per-person classifier trained with pure NumPy.

    Input vector: 7 DeepFace emotion probabilities + frame quality score.
    Output: 7 emotion probabilities in EMOTION_ORDER.
    """

    def __init__(self, weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        self.weights = weights
        self.bias = bias

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 700,
        lr: float = 0.18,
        l2: float = 0.001,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        c = len(EMOTION_ORDER)
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0, 0.02, size=(d, c))
        self.bias = np.zeros(c, dtype=np.float64)
        Y = np.zeros((n, c), dtype=np.float64)
        Y[np.arange(n), y] = 1.0
        last_loss = 0.0
        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            probs = self._softmax(logits)
            loss = -float(np.mean(np.sum(Y * np.log(np.maximum(probs, 1e-12)), axis=1)))
            loss += float(0.5 * l2 * np.sum(self.weights * self.weights))
            grad = (probs - Y) / max(1, n)
            grad_w = X.T @ grad + l2 * self.weights
            grad_b = grad.sum(axis=0)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
            last_loss = loss
        pred = self.predict_proba(X).argmax(axis=1)
        acc = float(np.mean(pred == y)) if len(y) else 0.0
        return {"loss": round(last_loss, 4), "train_accuracy": round(acc, 4), "epochs": epochs}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("adapter 尚未训练")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._softmax(X @ self.weights + self.bias)

    def save(self, path: Path, metadata: Dict[str, Any]) -> None:
        if self.weights is None or self.bias is None:
            raise ValueError("adapter 尚未训练，无法保存")
        np.savez(path, weights=self.weights, bias=self.bias, metadata=json.dumps(metadata, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> Tuple["SoftmaxEmotionAdapter", Dict[str, Any]]:
        data = np.load(path, allow_pickle=False)
        metadata_raw = str(data.get("metadata", "{}"))
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
        return cls(weights=data["weights"], bias=data["bias"]), metadata


class IdentityManager:
    """Local face profile database + personalized adapter manager."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._adapter_cache: Dict[str, Tuple[float, SoftmaxEmotionAdapter, Dict[str, Any]]] = {}

    def _profile_dir(self, person_id: str) -> Path:
        return self.root / person_id

    def _profile_path(self, person_id: str) -> Path:
        return self._profile_dir(person_id) / "profile.json"

    def _embeddings_path(self, person_id: str) -> Path:
        return self._profile_dir(person_id) / "embeddings.npy"

    def _samples_path(self, person_id: str) -> Path:
        return self._profile_dir(person_id) / "emotion_samples.csv"

    def _adapter_path(self, person_id: str) -> Path:
        return self._profile_dir(person_id) / "adapter.npz"

    def _adapter_metadata(self, person_id: str) -> Optional[Dict[str, Any]]:
        path = self._adapter_path(person_id)
        if not path.exists():
            return None
        try:
            _, metadata = SoftmaxEmotionAdapter.load(path)
            return metadata
        except Exception:
            return None

    def sample_summary(self, person_id: str) -> Dict[str, Any]:
        """Return label distribution and reliability hints for a profile."""
        rows = self._load_samples(person_id)
        counts = Counter(str(row.get("label", "")) for row in rows if row.get("label") in EMOTION_ORDER)
        distribution = {emotion: int(counts.get(emotion, 0)) for emotion in EMOTION_ORDER}
        present_counts = [count for count in distribution.values() if count > 0]
        total = int(sum(distribution.values()))
        min_present = int(min(present_counts)) if present_counts else 0
        warnings: List[str] = []
        if total < 20:
            warnings.append("总样本数少于 20：个人适配器仅适合课堂演示，不建议强解释性能。")
        if present_counts and min_present < 5:
            warnings.append("部分已采集类别少于 5 条：该类别容易过拟合。")
        if len(present_counts) < 2:
            warnings.append("少于 2 个表情类别：无法训练可靠的区分模型。")
        return {
            "total": total,
            "label_distribution": distribution,
            "num_present_labels": len(present_counts),
            "min_samples_per_present_label": min_present,
            "recommended_min_total": 20,
            "recommended_min_per_label": 5,
            "warnings": warnings,
        }

    def list_profiles(self) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []
        for profile_path in sorted(self.root.glob("*/profile.json")):
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                person_id = profile.get("person_id", profile_path.parent.name)
                profile["num_embeddings"] = int(self._load_embeddings(person_id).shape[0])
                profile["num_samples"] = len(self._load_samples(person_id))
                profile["adapter_available"] = self._adapter_path(person_id).exists()
                profile["sample_summary"] = self.sample_summary(person_id)
                profile["adapter_metadata"] = self._adapter_metadata(person_id)
                profiles.append(profile)
            except Exception:
                continue
        return profiles

    def create_profile(self, name: str) -> Dict[str, Any]:
        name = name.strip() or "未命名人物"
        person_id = f"{slugify_name(name)}_{now_slug()}"
        profile_dir = self._profile_dir(person_id)
        (profile_dir / "face_images").mkdir(parents=True, exist_ok=True)
        profile = {
            "person_id": person_id,
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "face_model": "Facenet512",
            "adapter_type": "numpy_softmax_emotion_adapter",
            "note": "Local-only biometric profile. Do not upload without consent.",
        }
        self._profile_path(person_id).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile

    def load_profile(self, person_id: str) -> Dict[str, Any]:
        path = self._profile_path(person_id)
        if not path.exists():
            raise FileNotFoundError(f"人物档案不存在: {person_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_profile(self, person_id: str, profile: Dict[str, Any]) -> None:
        self._profile_path(person_id).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_embeddings(self, person_id: str) -> np.ndarray:
        path = self._embeddings_path(person_id)
        if not path.exists():
            return np.empty((0, 0), dtype=np.float64)
        return np.load(path)

    def _save_embeddings(self, person_id: str, embeddings: np.ndarray) -> None:
        np.save(self._embeddings_path(person_id), np.asarray(embeddings, dtype=np.float64))

    def _prepare_profile_face_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return a clean face crop for profile enrollment.

        The browser normally sends a crop based on the latest detected bbox.  This
        backend fallback still tries to detect and keep the largest face, so a full
        screenshot will not be embedded as if the whole frame were a face.
        """
        try:
            from deepface import DeepFace

            extracted_faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
            )
            candidates: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
            for extracted in extracted_faces:
                area = extracted.get("facial_area") or {}
                face_arr = convert_face_image(extracted.get("face"))
                if face_arr is None:
                    continue
                w = float(area.get("w", face_arr.shape[1]))
                h = float(area.get("h", face_arr.shape[0]))
                confidence = float(extracted.get("confidence", 0.0) or 0.0)
                score = max(1.0, w * h) * max(0.15, confidence)
                candidates.append((score, face_arr, area))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                _, best_face, best_area = candidates[0]
                return best_face, {"used_detector": "opencv", "area": best_area}
        except Exception:
            pass
        return img, {"used_detector": "skip_fallback", "area": {"x": 0, "y": 0, "w": int(img.shape[1]), "h": int(img.shape[0])}}

    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            from deepface import DeepFace

            reps = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                silent=True,
            )
            if isinstance(reps, list) and reps:
                emb = reps[0].get("embedding")
                if emb is not None:
                    return np.asarray(emb, dtype=np.float64)
        except Exception:
            return None
        return None

    def add_face_image(self, person_id: str, image_base64: str) -> Dict[str, Any]:
        profile = self.load_profile(person_id)
        img = decode_base64_image(image_base64)
        face_img, crop_info = self._prepare_profile_face_image(img)
        emb = self.extract_embedding(face_img)
        if emb is None:
            raise ValueError("当前截图无法提取人脸 embedding，请让人脸更清晰后重试")
        face_dir = self._profile_dir(person_id) / "face_images"
        face_dir.mkdir(parents=True, exist_ok=True)
        image_path = face_dir / f"face_{now_slug()}_{int(time.time() * 1000) % 100000}.jpg"
        cv2.imwrite(str(image_path), face_img)
        old = self._load_embeddings(person_id)
        embeddings = emb.reshape(1, -1) if old.size == 0 else np.vstack([old, emb.reshape(1, -1)])
        self._save_embeddings(person_id, embeddings)
        profile["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._save_profile(person_id, profile)
        return {
            "person_id": person_id,
            "name": profile.get("name"),
            "num_embeddings": int(embeddings.shape[0]),
            "crop_info": crop_info,
        }

    def has_any_embedding(self) -> bool:
        for profile in self.list_profiles():
            if self._load_embeddings(profile["person_id"]).size > 0:
                return True
        return False

    def match_embedding(self, embedding: Optional[np.ndarray], threshold: float = 0.35) -> Dict[str, Any]:
        if embedding is None:
            return {"person_id": None, "person_name": None, "identity_distance": None, "identity_confidence": 0.0}
        best: Tuple[Optional[str], Optional[str], float] = (None, None, 999.0)
        for profile in self.list_profiles():
            person_id = profile["person_id"]
            embeddings = self._load_embeddings(person_id)
            if embeddings.size == 0:
                continue
            distances = [cosine_distance(embedding, ref) for ref in embeddings]
            dist = float(min(distances))
            if dist < best[2]:
                best = (person_id, profile.get("name"), dist)
        if best[0] is None or best[2] > threshold:
            return {
                "person_id": None,
                "person_name": None,
                "identity_distance": round(best[2], 4) if best[2] < 999 else None,
                "identity_confidence": 0.0,
            }
        confidence = max(0.0, min(100.0, (1.0 - best[2] / max(threshold, 1e-6)) * 100.0))
        return {
            "person_id": best[0],
            "person_name": best[1],
            "identity_distance": round(best[2], 4),
            "identity_confidence": round(confidence, 1),
        }

    def _load_samples(self, person_id: str) -> List[Dict[str, Any]]:
        path = self._samples_path(person_id)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def add_sample(self, person_id: str, label: str, emotion_probs: Dict[str, Any], quality_score: float = 1.0) -> Dict[str, Any]:
        if label not in EMOTION_ORDER:
            raise ValueError(f"不支持的表情标签: {label}")
        self.load_profile(person_id)
        probs = normalize_emotion_scores(emotion_probs)
        path = self._samples_path(person_id)
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            fieldnames = ["created_at", "label", "quality_score"] + EMOTION_ORDER
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            row = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "label": label,
                "quality_score": round(float(quality_score), 4),
            }
            row.update({emotion: round(float(probs[emotion]), 8) for emotion in EMOTION_ORDER})
            writer.writerow(row)
        summary = self.sample_summary(person_id)
        return {"person_id": person_id, "num_samples": summary["total"], "label": label, "sample_summary": summary}

    def _leave_one_out_accuracy(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Small-sample validation for the per-person adapter.

        This is intentionally simple and deterministic.  For classroom-sized
        calibration sets it gives a more honest signal than train accuracy.  If
        the profile becomes very large, we skip it to keep the UI responsive.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = int(len(y))
        if n < 3 or n > 150:
            return None
        correct = 0
        for holdout in range(n):
            mask = np.ones(n, dtype=bool)
            mask[holdout] = False
            adapter = SoftmaxEmotionAdapter()
            adapter.fit(X[mask], y[mask], epochs=260, lr=0.16, l2=0.002)
            pred = int(adapter.predict_proba(X[holdout]).argmax(axis=1)[0])
            correct += int(pred == int(y[holdout]))
        return round(float(correct / n), 4)

    def train_adapter(self, person_id: str) -> Dict[str, Any]:
        self.load_profile(person_id)
        rows = self._load_samples(person_id)
        if len(rows) < 6:
            raise ValueError("样本太少：至少建议采集 6 条手动标注样本")
        labels = [row["label"] for row in rows]
        distinct = sorted(set(labels))
        if len(distinct) < 2:
            raise ValueError("至少需要 2 个不同表情类别，才能训练个体化适配器")
        X = []
        y = []
        for row in rows:
            probs = {emotion: float(row.get(emotion, 0.0)) for emotion in EMOTION_ORDER}
            X.append(emotion_vector(normalize_emotion_scores(probs), float(row.get("quality_score", 1.0))))
            y.append(EMOTION_ORDER.index(row["label"]))
        adapter = SoftmaxEmotionAdapter()
        metrics = adapter.fit(np.asarray(X), np.asarray(y))
        sample_summary = self.sample_summary(person_id)
        loo_acc = self._leave_one_out_accuracy(np.asarray(X), np.asarray(y))
        validation_warnings = list(sample_summary.get("warnings", []))
        if loo_acc is not None and loo_acc < 0.45:
            validation_warnings.append("留一法准确率偏低：建议继续补充更清晰、更多类别的手动样本。")
        metadata = {
            "person_id": person_id,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "num_samples": len(rows),
            "labels": distinct,
            "emotion_order": EMOTION_ORDER,
            "label_distribution": sample_summary["label_distribution"],
            "min_samples_per_present_label": sample_summary["min_samples_per_present_label"],
            "leave_one_out_accuracy": loo_acc,
            "validation_warnings": validation_warnings,
            **metrics,
        }
        adapter.save(self._adapter_path(person_id), metadata)
        self._adapter_cache.pop(person_id, None)
        return metadata

    def predict_adapter(
        self,
        person_id: Optional[str],
        emotion_probs: Dict[str, float],
        quality_score: float = 1.0,
    ) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
        if not person_id:
            return None
        path = self._adapter_path(person_id)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cached = self._adapter_cache.get(person_id)
        if cached is None or cached[0] != mtime:
            adapter, metadata = SoftmaxEmotionAdapter.load(path)
            self._adapter_cache[person_id] = (mtime, adapter, metadata)
        _, adapter, metadata = self._adapter_cache[person_id]
        X = emotion_vector(normalize_emotion_scores(emotion_probs), quality_score)
        probs_arr = adapter.predict_proba(X)[0]
        probs = {emotion: float(probs_arr[i]) for i, emotion in enumerate(EMOTION_ORDER)}
        return probs, metadata


class SessionEmotionProcessor:
    """Stateful per-WebSocket processor.

    It attaches track ids, quality scores, segment ids, optional identity matches,
    personalized adapter outputs, and causal Gaussian-smoothed probabilities.
    """

    def __init__(self, identity_manager: IdentityManager):
        self.identity_manager = identity_manager
        self.quality_estimator = QualityEstimator()
        self.smoother = CausalGaussianSmoother()
        self.segment_id = 0
        self.no_face_streak = 0
        self.prev_signature: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None
        self.prev_tracks: Dict[str, Dict[str, Any]] = {}
        self.next_track_id = 1
        self.prev_known_person: Optional[str] = None

    def _new_segment(self) -> None:
        self.segment_id += 1
        self.smoother.reset()
        self.prev_tracks.clear()
        self.prev_known_person = None

    def _assign_track(self, area: Dict[str, Any]) -> str:
        best_id, best_iou = None, 0.0
        for track_id, info in self.prev_tracks.items():
            score = bbox_iou(area, info.get("area"))
            if score > best_iou:
                best_id, best_iou = track_id, score
        if best_id is not None and best_iou >= 0.18:
            return best_id
        track_id = f"track_{self.next_track_id}"
        self.next_track_id += 1
        return track_id

    def note_no_face(self, frame: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
        self.no_face_streak += 1
        if self.no_face_streak >= int(cfg.get("continuity_no_face_gap", 5)):
            self._new_segment()
            self.no_face_streak = 0
        self.prev_signature = frame_signature(frame)
        self.prev_timestamp = time.time()
        return {"segment_id": self.segment_id, "no_face_streak": self.no_face_streak}

    def process_faces(self, frame: np.ndarray, faces: List[Dict[str, Any]], cfg: Dict[str, Any], frame_index: int) -> List[Dict[str, Any]]:
        timestamp = time.time()
        expected_gap = max(0.05, float(cfg.get("analyze_interval", 5)) / max(1.0, float(cfg.get("fps", 10))))
        delta = frame_delta(self.prev_signature, frame)
        time_gap = (timestamp - self.prev_timestamp) if self.prev_timestamp else 0.0
        if self.prev_timestamp and time_gap > float(cfg.get("continuity_max_gap_factor", 3.0)) * expected_gap:
            self._new_segment()
        elif delta > float(cfg.get("continuity_frame_diff_threshold", 0.48)):
            self._new_segment()
        elif self.no_face_streak >= int(cfg.get("continuity_no_face_gap", 5)):
            self._new_segment()

        self.no_face_streak = 0
        processed: List[Dict[str, Any]] = []
        new_tracks: Dict[str, Dict[str, Any]] = {}
        current_known_person: Optional[str] = None

        for face in faces:
            area = face.get("area") or {}
            track_id = self._assign_track(area)
            previous_area = self.prev_tracks.get(track_id, {}).get("area")
            quality_score, quality_details = self.quality_estimator.estimate(
                frame=frame,
                area=area,
                face_confidence=float(face.get("face_conf", 0.0)),
                previous_area=previous_area,
            )
            raw_probs = normalize_emotion_scores(face.get("emotion_probs", {}))
            raw_emotion, raw_conf = dominant_from_probs(raw_probs)

            person_match: Dict[str, Any] = {
                "person_id": None,
                "person_name": None,
                "identity_distance": None,
                "identity_confidence": 0.0,
            }
            face_crop = crop_bbox(frame, area)
            if bool(cfg.get("identity_enabled", True)) and face_crop is not None and self.identity_manager.has_any_embedding():
                embedding = self.identity_manager.extract_embedding(face_crop)
                person_match = self.identity_manager.match_embedding(
                    embedding, threshold=float(cfg.get("identity_threshold", 0.35))
                )
                if person_match.get("person_id"):
                    current_known_person = person_match.get("person_id")

            adapted_probs = raw_probs
            adapter_info: Optional[Dict[str, Any]] = None
            adapter_prediction: Optional[Dict[str, Any]] = None
            if bool(cfg.get("adapter_enabled", True)) and person_match.get("person_id"):
                adapter_result = self.identity_manager.predict_adapter(
                    person_match.get("person_id"), raw_probs, quality_score
                )
                if adapter_result is not None:
                    personal_probs, adapter_info = adapter_result
                    blend = max(0.0, min(0.85, float(cfg.get("adapter_blend", 0.55))))
                    adapted_probs = {
                        emotion: (1.0 - blend) * raw_probs[emotion] + blend * personal_probs[emotion]
                        for emotion in EMOTION_ORDER
                    }
                    adapter_emotion, adapter_conf = dominant_from_probs(personal_probs)
                    adapter_prediction = {
                        "emotion": adapter_emotion,
                        "confidence": round(adapter_conf * 100.0, 1),
                        "blend": round(blend, 2),
                        "num_samples": adapter_info.get("num_samples") if adapter_info else None,
                    }

            key = f"person:{person_match.get('person_id')}" if person_match.get("person_id") else f"track:{track_id}"
            if bool(cfg.get("smoothing_enabled", True)):
                smoothed_probs = self.smoother.update(
                    key=key,
                    probs=adapted_probs,
                    quality_score=quality_score if bool(cfg.get("quality_penalty_enabled", True)) else 1.0,
                    frame_index=frame_index,
                    segment_id=self.segment_id,
                    timestamp=timestamp,
                    window=int(cfg.get("smoothing_window", 5)),
                    sigma=float(cfg.get("smoothing_sigma", 2.0)),
                )
            else:
                smoothed_probs = adapted_probs

            final_emotion, final_conf = dominant_from_probs(smoothed_probs)
            face.update(
                {
                    "track_id": track_id,
                    "segment_id": self.segment_id,
                    "raw_emotion": raw_emotion,
                    "raw_e_conf": round(raw_conf * 100.0, 1),
                    "raw_emotion_probs": probs_to_percent_dict(raw_probs),
                    "emotion_probs": probs_to_percent_dict(smoothed_probs),
                    "emotion": final_emotion,
                    "e_conf": round(final_conf * 100.0, 1),
                    "quality_score": quality_score,
                    "quality_details": quality_details,
                    "person_id": person_match.get("person_id"),
                    "person_name": person_match.get("person_name"),
                    "identity_distance": person_match.get("identity_distance"),
                    "identity_confidence": person_match.get("identity_confidence", 0.0),
                    "personalized": adapter_prediction is not None,
                    "adapter_prediction": adapter_prediction,
                    "smoothing": {
                        "enabled": bool(cfg.get("smoothing_enabled", True)),
                        "window": int(cfg.get("smoothing_window", 5)),
                        "sigma": float(cfg.get("smoothing_sigma", 2.0)),
                        "quality_penalty": bool(cfg.get("quality_penalty_enabled", True)),
                    },
                    "continuity": {
                        "frame_delta": round(delta, 3),
                        "time_gap": round(time_gap, 3),
                    },
                }
            )
            processed.append(face)
            new_tracks[track_id] = {"area": area, "last_seen": frame_index}

        if current_known_person and self.prev_known_person and current_known_person != self.prev_known_person:
            self._new_segment()
            for face in processed:
                face["segment_id"] = self.segment_id
                face.setdefault("continuity", {})["identity_cut"] = True
        if current_known_person:
            self.prev_known_person = current_known_person

        self.prev_tracks = new_tracks
        self.prev_signature = frame_signature(frame)
        self.prev_timestamp = timestamp
        return processed
