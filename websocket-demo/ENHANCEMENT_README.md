# Enhanced FER Demo: Gaussian Smoothing + Personalized Emotion Adapter

本仓库在原有 `websocket-demo` 实时 UI 的基础上完成了两类增强：

1. **Temporal Emotion Smoother**：高斯时序平滑、坏帧质量惩罚、视频/流连续性断裂截断。
2. **Personalized Emotion Adapter**：本地人脸建档、身份匹配、手动标注样本、特定人物表情小模型。

## 1. 实时摄像头：高斯平滑与坏帧惩罚

后端核心文件：

```plain
websocket-demo/emotion_enhance.py
websocket-demo/server.py
websocket-demo/index.html
```

实时流中，系统只能看到过去帧，因此采用因果高斯平滑：

```plain
P_smooth(t) = Σ_i w(i,t) P(i) / Σ_i w(i,t),  i ∈ [t-K, t]
w(i,t) = exp(-(i-t)^2 / (2σ^2)) · q_i
```

其中 `q_i` 是质量分数，由以下项组成：

```plain
q = 0.35*q_face + 0.20*q_area + 0.20*q_blur + 0.15*q_center + 0.10*q_bbox_stability
```

含义：

- `q_face`：DeepFace/检测器的人脸置信度。
- `q_area`：人脸区域是否足够大。
- `q_blur`：Laplacian variance 估计的清晰度。
- `q_center`：人脸是否偏离画面中心。
- `q_bbox_stability`：当前 bbox 与上一帧 bbox 的 IoU 连续性。

断裂截断机制：

- 连续无脸超过 `continuity_no_face_gap`。
- 两次分析之间时间间隔过大。
- 当前画面与上一分析帧的缩略图差异超过 `continuity_frame_diff_threshold`。
- 识别到的人物身份发生变化。

发生断裂后，`segment_id += 1`，平滑缓存清空，避免跨镜头、离屏重入、切换人物时错误平滑。

## 2. 离线视频：双向高斯平滑

新增脚本：

```bash
python websocket-demo/analyze_video_offline.py \
  --input demo.mp4 \
  --output outputs/demo_result.json \
  --every 5 \
  --window 5 \
  --sigma 2.0 \
  --detector_backend retinaface
```

离线视频可以看到未来帧，因此使用双向窗口：

```plain
i ∈ [t-K, ..., t-1, t, t+1, ..., t+K]
```

但仍然不会跨越 `segment_id` 进行平滑。

## 3. 个体化建档与表情适配器

UI 新增了 **个体化表情建档** 区域：

1. 输入人物名称，点击“创建档案”。
2. 开启摄像头，保持人脸清晰，点击“采集当前画面做人脸建档”。前端会优先按当前检测框裁剪人脸，后端也会二次检测最大人脸，避免把整张背景图当作人脸 embedding。建议每人采集 3~5 张。
3. 选择手动表情标签，例如 `happy` / `neutral` / `angry`。
4. 点击“保存当前识别为标注样本”。建议至少 2 个类别，每类数条样本。
5. 点击“训练个人适配器”。
6. 后续识别到该人物后，系统会融合个人适配器输出与 DeepFace 原始输出。

数据保存位置：

```plain
websocket-demo/data/face_profiles/<person_id>/
  profile.json
  face_images/*.jpg
  embeddings.npy
  emotion_samples.csv
  adapter.npz
```

个体化适配器没有引入 sklearn / PyTorch 等新依赖，而是用 NumPy 实现了一个轻量 softmax 分类器：

```plain
输入 x = [7维 DeepFace emotion probability + quality_score]
输出 y = 7类个体化 emotion probability
```

最终融合：

```plain
P_final = (1 - λ) * P_deepface + λ * P_personal_adapter
```

其中 `λ` 可在 UI 配置中通过“适配器融合 λ”调整。

## 4. 运行

```bash
cd websocket-demo
python server.py
```

打开浏览器访问：

```plain
http://localhost:8000
```

注意：原 demo 依赖 `ffmpeg` 进行 WebM 流解码；系统里需要能直接运行 `ffmpeg -version`。

## 5. 隐私说明

人脸 embedding 和截图是生物特征数据。本实现默认全部保存在本地 `websocket-demo/data/face_profiles`，不上传远端。真实使用时请确保被采集者知情同意，并避免把该目录提交到公开仓库。

## 6. 额外鲁棒性修正

- 实时识别与离线视频分析都会过滤“置信度极低且几乎覆盖整帧”的 fallback 伪人脸，避免 `enforce_detection=False` 时将无脸画面误当作人脸继续输出表情。
- 人脸建档时保存的是裁剪后的人脸图，而不是完整摄像头画面；这能显著提高后续身份匹配与个体化适配器的稳定性。
- 如果检测器没有返回可信置信度，质量分中的 `q_face` 会被保守降权，而不会默认视为高质量帧。

## 7. 实验结果自动导出

新增脚本：

```bash
python websocket-demo/export_experiment_report.py \
  --input outputs/demo_result.json \
  --output-dir outputs/demo_report
```

该脚本会从 `analyze_video_offline.py` 生成的 JSON 中导出：

```plain
summary.json
summary.csv
frame_records.csv
emotion_curve.png
```

核心指标包括：

```plain
raw_emotion_switch_count
smooth_emotion_switch_count
switch_reduction_percent
no_face_rate
average_quality_score
segment_count
```

这些结果可以直接放进研究报告的“实验结果与分析”部分。

## 8. 连续采样与 adapter 验证指标

前端“个体化表情建档”区域新增：

```plain
连续采样时长 / 秒
最低质量分
连续采样 5 秒
样本分布与验证指标显示
```

训练个人适配器后，系统会显示：

```plain
train_accuracy
leave_one_out_accuracy
label_distribution
validation_warnings
```

其中 `leave_one_out_accuracy` 比单纯训练集准确率更适合小样本演示，用于提醒 adapter 是否明显过拟合。若总样本少于 20 或某些类别少于 5 条，UI 会给出可靠性警告。

完整说明见：

```plain
websocket-demo/PROJECT_MODIFICATION_REPORT.md
```
