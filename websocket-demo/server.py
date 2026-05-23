import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import uvicorn
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from emotion_enhance import (
    EMOTION_ORDER,
    IdentityManager,
    SessionEmotionProcessor,
    dominant_from_probs,
    normalize_emotion_scores,
    probs_to_percent_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SUPPORTED_BACKENDS = {"opencv", "retinaface", "mtcnn", "mediapipe", "centerface", "skip"}
DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "width": 640,
    "height": 480,
    "fps": 15,
    "analyze_interval": 2,
    "detector_backend": "opencv",
    "align": False,
    "chunk_ms": 80,
    "video_bits_per_second": 600000,
    "timeout_seconds": 45,
    "smoothing_enabled": True,
    "smoothing_window": 5,
    "smoothing_sigma": 2.0,
    "quality_penalty_enabled": True,
    "continuity_no_face_gap": 5,
    "continuity_max_gap_factor": 3.0,
    "continuity_frame_diff_threshold": 0.48,
    "identity_enabled": True,
    "identity_threshold": 0.35,
    "adapter_enabled": True,
    "adapter_blend": 0.55,
}

config_lock = threading.Lock()
stats_lock = threading.Lock()
runtime_config: Dict[str, Any] = DEFAULT_RUNTIME_CONFIG.copy()
active_decoders: Dict[int, "FFmpegWebMStreamDecoder"] = {}
identity_manager = IdentityManager(Path(__file__).parent / "data" / "face_profiles")


def resolve_ffmpeg_exe() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg is required for browser WebM decoding. Install ffmpeg or run "
            "`python -m pip install imageio-ffmpeg`."
        ) from exc


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def new_stats() -> Dict[str, Any]:
    return {
        "session_started_at": time.time(),
        "total_chunks_received": 0,
        "total_frames_received": 0,
        "analyzed_frames": 0,
        "successful_analyses": 0,
        "no_face_events": 0,
        "results_sent": 0,
        "last_result_at": None,
        "last_result": None,
        "emotion_distribution": {emotion: 0 for emotion in EMOTION_ORDER},
        "recent_events": deque(maxlen=18),
    }


runtime_stats = new_stats()


def get_runtime_config() -> Dict[str, Any]:
    with config_lock:
        return dict(runtime_config)


def record_event(level: str, message: str, client_id: Optional[int] = None) -> None:
    event = {
        "timestamp": now_iso(),
        "level": level,
        "message": message,
    }
    if client_id is not None:
        event["client_id"] = client_id
    with stats_lock:
        runtime_stats["recent_events"].appendleft(event)


def snapshot_stats() -> Dict[str, Any]:
    with stats_lock:
        uptime = round(time.time() - runtime_stats["session_started_at"], 1)
        return {
            "session_started_at": runtime_stats["session_started_at"],
            "uptime_seconds": uptime,
            "active_clients": len(active_decoders),
            "total_chunks_received": runtime_stats["total_chunks_received"],
            "total_frames_received": runtime_stats["total_frames_received"],
            "analyzed_frames": runtime_stats["analyzed_frames"],
            "successful_analyses": runtime_stats["successful_analyses"],
            "no_face_events": runtime_stats["no_face_events"],
            "results_sent": runtime_stats["results_sent"],
            "last_result_at": runtime_stats["last_result_at"],
            "last_result": runtime_stats["last_result"],
            "emotion_distribution": dict(runtime_stats["emotion_distribution"]),
            "recent_events": list(runtime_stats["recent_events"]),
        }


def reset_stats() -> None:
    global runtime_stats
    with stats_lock:
        runtime_stats = new_stats()
    record_event("info", "统计信息已重置")


def normalize_face_area(area: Dict[str, Any]) -> Dict[str, int]:
    return {
        "x": int(area.get("x", 0)),
        "y": int(area.get("y", 0)),
        "w": int(area.get("w", 0)),
        "h": int(area.get("h", 0)),
    }


def dominant_emotion_from_faces(faces: list[Dict[str, Any]]) -> str:
    counter = Counter(face["emotion"] for face in faces if face.get("emotion"))
    if not counter:
        return "none"
    return counter.most_common(1)[0][0]


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


def analyze_detected_faces(frame: np.ndarray, detector_backend: str, align: bool) -> list[Dict[str, Any]]:
    extracted_faces = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=align,
    )

    analyzed_faces: list[Dict[str, Any]] = []
    for idx, extracted_face in enumerate(extracted_faces):
        try:
            face_image = convert_face_image(extracted_face.get("face"))
            if face_image is None:
                continue

            demographies = DeepFace.analyze(
                img_path=face_image,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            analysis = demographies[0] if isinstance(demographies, list) else demographies
            area = normalize_face_area(extracted_face.get("facial_area") or analysis.get("region") or {})
            face_conf = float(extracted_face.get("confidence", analysis.get("face_confidence", 0.0)) or 0.0)
            frame_h, frame_w = frame.shape[:2]
            area_ratio = (max(0, area.get("w", 0)) * max(0, area.get("h", 0))) / max(1, frame_w * frame_h)
            looks_like_fallback = area_ratio > 0.82 and face_conf <= 0.05
            too_tiny = area_ratio < 0.002
            if looks_like_fallback or too_tiny:
                logging.debug(
                    "跳过低可信伪人脸: area_ratio=%.3f confidence=%.3f fallback=%s tiny=%s",
                    area_ratio,
                    face_conf,
                    looks_like_fallback,
                    too_tiny,
                )
                continue

            dominant_emotion = str(analysis.get("dominant_emotion", "neutral"))
            emotion_scores = analysis.get("emotion", {}) or {}
            emotion_probs = normalize_emotion_scores(emotion_scores)
            dominant_emotion, dominant_conf = dominant_from_probs(emotion_probs)

            analyzed_faces.append(
                {
                    "id": idx,
                    "area": area,
                    "age": None,
                    "gender": None,
                    "g_conf": 0.0,
                    "emotion": dominant_emotion,
                    "e_conf": float(round(dominant_conf * 100.0, 1)),
                    "raw_emotion": dominant_emotion,
                    "raw_e_conf": float(round(dominant_conf * 100.0, 1)),
                    "emotion_probs": probs_to_percent_dict(emotion_probs),
                    "raw_emotion_probs": probs_to_percent_dict(emotion_probs),
                    "face_conf": float(round(face_conf, 3)),
                }
            )
        except Exception as exc:
            logging.error("单张人脸分析失败: %s", exc)

    return analyzed_faces


def build_summary(frame_count: int, faces: list[Dict[str, Any]], analysis_duration_ms: float) -> Dict[str, Any]:
    quality_scores = [float(face["quality_score"]) for face in faces if face.get("quality_score") is not None]
    return {
        "frame_index": frame_count,
        "faces_detected": len(faces),
        "dominant_emotion": dominant_emotion_from_faces(faces),
        "average_age": None,
        "average_quality": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None,
        "analysis_duration_ms": round(analysis_duration_ms, 1),
    }


def build_payload(
    payload_type: str,
    frame_count: int,
    frame: np.ndarray,
    faces: list[Dict[str, Any]],
    analysis_duration_ms: float,
    message: str = "",
) -> Dict[str, Any]:
    return {
        "type": payload_type,
        "timestamp": now_iso(),
        "message": message,
        "source": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "fps": get_runtime_config()["fps"],
        },
        "summary": build_summary(frame_count, faces, analysis_duration_ms),
        "count": len(faces),
        "faces": faces,
        "stats": snapshot_stats(),
        "config": get_runtime_config(),
    }


def update_stats_after_analysis(payload: Dict[str, Any]) -> None:
    with stats_lock:
        runtime_stats["results_sent"] += 1
        runtime_stats["last_result_at"] = payload["timestamp"]
        runtime_stats["last_result"] = {
            "type": payload["type"],
            "count": payload["count"],
            "dominant_emotion": payload["summary"]["dominant_emotion"],
            "analysis_duration_ms": payload["summary"]["analysis_duration_ms"],
            "timestamp": payload["timestamp"],
        }
        if payload["type"] == "result":
            runtime_stats["successful_analyses"] += 1
            for face in payload["faces"]:
                emotion = face.get("emotion")
                if emotion in runtime_stats["emotion_distribution"]:
                    runtime_stats["emotion_distribution"][emotion] += 1
        elif payload["type"] == "no_face":
            runtime_stats["no_face_events"] += 1


class RuntimeConfigPayload(BaseModel):
    width: Optional[int] = Field(default=None, ge=320, le=1280)
    height: Optional[int] = Field(default=None, ge=240, le=960)
    fps: Optional[int] = Field(default=None, ge=5, le=30)
    analyze_interval: Optional[int] = Field(default=None, ge=1, le=20)
    detector_backend: Optional[str] = None
    align: Optional[bool] = None
    chunk_ms: Optional[int] = Field(default=None, ge=60, le=1000)
    video_bits_per_second: Optional[int] = Field(default=None, ge=200000, le=5000000)
    timeout_seconds: Optional[int] = Field(default=None, ge=10, le=120)
    smoothing_enabled: Optional[bool] = None
    smoothing_window: Optional[int] = Field(default=None, ge=1, le=20)
    smoothing_sigma: Optional[float] = Field(default=None, ge=0.3, le=10.0)
    quality_penalty_enabled: Optional[bool] = None
    continuity_no_face_gap: Optional[int] = Field(default=None, ge=1, le=60)
    continuity_max_gap_factor: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    continuity_frame_diff_threshold: Optional[float] = Field(default=None, ge=0.05, le=1.0)
    identity_enabled: Optional[bool] = None
    identity_threshold: Optional[float] = Field(default=None, ge=0.05, le=1.5)
    adapter_enabled: Optional[bool] = None
    adapter_blend: Optional[float] = Field(default=None, ge=0.0, le=0.85)


class ProfileCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ProfileFacePayload(BaseModel):
    image_base64: str = Field(min_length=20)


class AdapterSamplePayload(BaseModel):
    label: str
    emotion_probs: Dict[str, float]
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)


class FFmpegWebMStreamDecoder:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 10, timeout_seconds: int = 45):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3
        self.timeout_seconds = timeout_seconds

        self.process: Optional[subprocess.Popen] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.write_lock = threading.Lock()
        self.last_frame_time = time.time()

        logging.info("创建 WebM 解码器: %sx%s@%sfps", width, height, fps)

    def start(self) -> bool:
        try:
            ffmpeg_exe = resolve_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-f",
                "webm",
                "-probesize",
                "32M",
                "-analyzeduration",
                "0",
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "pipe:1",
            ]

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )

            self.is_running = True
            threading.Thread(target=self._read_frames, daemon=True).start()
            threading.Thread(target=self._error_monitor, daemon=True).start()
            threading.Thread(target=self._health_monitor, daemon=True).start()
            logging.info("WebM 解码器启动成功")
            return True
        except Exception as exc:
            logging.error("FFmpeg 启动失败: %s", exc)
            return False

    def _read_frames(self) -> None:
        time.sleep(1)
        while self.is_running:
            try:
                if not self.process or not self.process.stdout:
                    continue

                frame_data = self.process.stdout.read(self.frame_size)
                if len(frame_data) == self.frame_size:
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(self.height, self.width, 3)
                    self.last_frame_time = time.time()
                    while self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                elif len(frame_data) == 0 and self.process.poll() is not None:
                    break
            except Exception as exc:
                logging.error("读帧异常: %s", exc)
                break

    def _error_monitor(self) -> None:
        while self.is_running:
            try:
                if self.process and self.process.stderr:
                    line = self.process.stderr.readline()
                    if line:
                        logging.debug("FFmpeg: %s", line.decode(errors="ignore").strip())
            except Exception:
                break

    def _health_monitor(self) -> None:
        time.sleep(5)
        while self.is_running:
            if self.process and self.process.poll() is not None:
                logging.error("FFmpeg 进程异常退出")
                self.stop()
                break
            if time.time() - self.last_frame_time > self.timeout_seconds:
                logging.error("超过 %s 秒无新帧，停止当前解码器", self.timeout_seconds)
                self.stop()
                break
            time.sleep(1)

    def write(self, webm_data: bytes) -> bool:
        if not self.is_healthy():
            return False
        with self.write_lock:
            try:
                if not self.process or not self.process.stdin:
                    return False
                self.process.stdin.write(webm_data)
                self.process.stdin.flush()
                return True
            except Exception as exc:
                logging.error("写入失败: %s", exc)
                self.stop()
                return False

    def read_frame(self) -> Optional[np.ndarray]:
        latest_frame = None
        try:
            while True:
                latest_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return latest_frame

    def is_healthy(self) -> bool:
        return bool(self.is_running and self.process and self.process.poll() is None)

    def stop(self) -> None:
        if not self.is_running:
            return

        self.is_running = False
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                logging.warning("FFmpeg 强制终止")
            except Exception as exc:
                logging.error("停止失败: %s", exc)

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        logging.info("FFmpeg 解码器已停止")


app = FastAPI(title="DeepFace WebSocket Demo")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    return get_runtime_config()


@app.post("/api/config")
async def update_config(payload: RuntimeConfigPayload) -> Dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    detector_backend = updates.get("detector_backend")
    if detector_backend and detector_backend not in SUPPORTED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"不支持的 detector_backend: {detector_backend}")

    with config_lock:
        runtime_config.update(updates)
        updated = dict(runtime_config)

    record_event("info", "运行配置已更新")
    return {"message": "配置已更新，新的视频参数将在重新启动识别后生效", "config": updated}


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    return snapshot_stats()


@app.post("/api/stats/reset")
async def reset_stats_endpoint() -> Dict[str, Any]:
    reset_stats()
    return {"message": "统计信息已重置", "stats": snapshot_stats()}


@app.get("/api/stats/export")
async def export_stats() -> Response:
    payload = {
        "config": get_runtime_config(),
        "stats": snapshot_stats(),
        "exported_at": now_iso(),
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="deepface-stats.json"'},
    )


@app.get("/api/profiles")
async def list_profiles() -> Dict[str, Any]:
    return {"profiles": identity_manager.list_profiles(), "emotion_order": EMOTION_ORDER}


@app.post("/api/profiles")
async def create_profile(payload: ProfileCreatePayload) -> Dict[str, Any]:
    profile = identity_manager.create_profile(payload.name)
    record_event("info", f"已创建人物档案: {profile['name']}")
    return {"message": "人物档案已创建", "profile": profile, "profiles": identity_manager.list_profiles()}


@app.post("/api/profiles/{person_id}/face")
async def add_profile_face(person_id: str, payload: ProfileFacePayload) -> Dict[str, Any]:
    try:
        result = identity_manager.add_face_image(person_id, payload.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record_event("info", f"已为 {result.get('name') or person_id} 添加人脸样本")
    return {"message": "人脸样本已加入建档", "result": result, "profiles": identity_manager.list_profiles()}


@app.post("/api/profiles/{person_id}/sample")
async def add_adapter_sample(person_id: str, payload: AdapterSamplePayload) -> Dict[str, Any]:
    if payload.label not in EMOTION_ORDER:
        raise HTTPException(status_code=400, detail=f"不支持的表情标签: {payload.label}")
    try:
        result = identity_manager.add_sample(
            person_id=person_id,
            label=payload.label,
            emotion_probs=payload.emotion_probs,
            quality_score=payload.quality_score,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record_event("info", f"已添加个体化表情样本: {payload.label}")
    return {"message": "手动标注样本已保存", "result": result, "profiles": identity_manager.list_profiles()}


@app.post("/api/profiles/{person_id}/train")
async def train_personal_adapter(person_id: str) -> Dict[str, Any]:
    try:
        metadata = identity_manager.train_adapter(person_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record_event("info", f"个体化表情适配器训练完成: {person_id}")
    return {"message": "个体化表情适配器训练完成", "adapter": metadata, "profiles": identity_manager.list_profiles()}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client_id = id(ws)
    cfg = get_runtime_config()
    decoder = FFmpegWebMStreamDecoder(
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg["fps"],
        timeout_seconds=cfg["timeout_seconds"],
    )
    session_processor = SessionEmotionProcessor(identity_manager)

    logging.info("客户端 %s 连接", client_id)
    record_event("info", "客户端已连接", client_id=client_id)

    if not decoder.start():
        await ws.send_json({"type": "error", "message": "解码器启动失败"})
        await ws.close()
        return

    active_decoders[client_id] = decoder
    await ws.send_json(
        {
            "type": "session",
            "message": "WebSocket 已连接，等待视频流数据",
            "config": cfg,
            "stats": snapshot_stats(),
        }
    )

    frame_count = 0
    try:
        while True:
            if not decoder.is_healthy():
                await ws.send_json({"type": "error", "message": "解码器异常，请重新连接"})
                break

            data = await ws.receive_bytes()
            if not data:
                continue

            with stats_lock:
                runtime_stats["total_chunks_received"] += 1

            if not decoder.write(data):
                await ws.send_json({"type": "error", "message": "视频数据写入失败"})
                break

            frame = decoder.read_frame()
            if frame is None:
                continue

            frame_count += 1
            with stats_lock:
                runtime_stats["total_frames_received"] += 1

            if frame_count % cfg["analyze_interval"] != 0:
                continue

            analysis_start = time.perf_counter()
            with stats_lock:
                runtime_stats["analyzed_frames"] += 1

            faces = analyze_detected_faces(frame, cfg["detector_backend"], cfg["align"])
            if faces:
                faces = session_processor.process_faces(frame, faces, cfg, frame_count)
            analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000

            if faces:
                payload = build_payload("result", frame_count, frame, faces, analysis_duration_ms)
                payload["segment_id"] = session_processor.segment_id
                update_stats_after_analysis(payload)
                payload["stats"] = snapshot_stats()
                await ws.send_json(payload)
                logging.info(
                    "检测到 %s 张人脸 - 主情绪: %s - 分析耗时: %.1fms",
                    payload["count"],
                    payload["summary"]["dominant_emotion"],
                    payload["summary"]["analysis_duration_ms"],
                )
            else:
                payload = build_payload(
                    "no_face",
                    frame_count,
                    frame,
                    [],
                    analysis_duration_ms,
                    message="当前分析帧未检测到清晰人脸",
                )
                payload["continuity"] = session_processor.note_no_face(frame, cfg)
                payload["segment_id"] = session_processor.segment_id
                update_stats_after_analysis(payload)
                payload["stats"] = snapshot_stats()
                await ws.send_json(payload)
    except WebSocketDisconnect:
        logging.info("客户端 %s 主动断开", client_id)
        record_event("warning", "客户端主动断开", client_id=client_id)
    except Exception as exc:
        logging.error("WebSocket 异常: %s", exc)
        record_event("error", f"WebSocket 异常: {exc}", client_id=client_id)
    finally:
        logging.info("客户端 %s 断开", client_id)
        if client_id in active_decoders:
            active_decoders[client_id].stop()
            active_decoders.pop(client_id, None)
        record_event("info", "客户端会话已结束", client_id=client_id)


if __name__ == "__main__":
    try:
        subprocess.run([resolve_ffmpeg_exe(), "-version"], check=True, capture_output=True)
        logging.info("FFmpeg 检查通过")
    except Exception as exc:
        logging.error("FFmpeg 未安装: %s", exc)
        raise SystemExit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
