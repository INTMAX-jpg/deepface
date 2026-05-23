# DeepFace 表情识别大作业增强说明

本文档说明本仓库相对原始 `websocket-demo` 做了哪些改动、如何运行，以及核心修改逻辑。适合直接放进课程报告/答辩材料的“工程实现说明”部分。

---

## 1. 原始系统基础

原始仓库已经具备一个基于浏览器 UI 的实时 DeepFace Demo：

```plain
浏览器摄像头
  → MediaRecorder/WebM 视频流
  → FastAPI WebSocket
  → FFmpeg 解码为 OpenCV 帧
  → DeepFace.extract_faces / DeepFace.analyze
  → 前端实时显示年龄、性别、表情、人脸框
```

本次改造没有推翻原有 UI，而是在原数据流上新增三个增强层：

```plain
DeepFace 单帧识别结果
  → 质量评估 QualityEstimator
  → 连续性检测 ContinuityDetector
  → 高斯时序平滑 GaussianTemporalSmoother
  → 个体化表情适配 PersonalizedEmotionAdapter
  → UI 展示与实验导出
```

---

## 2. 本次新增/修改文件

主要新增或修改如下：

```plain
websocket-demo/
  server.py                    # 后端接口、WebSocket 推理链路、人物档案 API
  index.html                   # 前端 UI：平滑参数、个体化建档、连续采样、验证指标
  emotion_enhance.py           # 核心增强模块：质量评估、平滑、人脸档案、个人适配器
  analyze_video_offline.py     # 离线视频分析：双向高斯平滑，输出 JSON
  export_experiment_report.py  # 新增：从 JSON 自动导出实验表格与曲线图
  ENHANCEMENT_README.md        # 简要增强说明
  PROJECT_MODIFICATION_REPORT.md  # 本说明文档
  data/face_profiles/          # 本地人物档案数据库，运行后自动创建
```

---

## 3. 功能一：高斯时序平滑

### 3.1 为什么要做

DeepFace 单帧表情识别在实时摄像头下容易出现跳变，例如：

```plain
neutral → sad → neutral → angry → neutral
```

这会导致 UI 的情绪水位计闪烁，影响演示稳定性。因此新增了时序平滑模块。

### 3.2 实时摄像头：因果高斯平滑

实时摄像头不能看到未来帧，所以对 t 时刻只使用当前帧和历史帧：

```plain
i ∈ [t-K, ..., t-2, t-1, t]
```

平滑公式：

```plain
P_smooth(t) = Σ_i w(i,t) P(i) / Σ_i w(i,t)

w(i,t) = exp(-(i-t)^2 / (2σ²)) × q_i
```

其中：

```plain
P(i) : 第 i 帧 DeepFace 输出的 7 类表情概率
q_i  : 第 i 帧质量分
K    : 平滑窗口大小
σ    : 高斯核标准差
```

### 3.3 离线视频：双向高斯平滑

如果输入是离线视频，可以看到未来帧，因此使用双向窗口：

```plain
i ∈ [t-K, ..., t-1, t, t+1, ..., t+K]
```

这比实时流更稳定，适合做实验对比和生成报告图。

---

## 4. 功能二：bad frame 质量惩罚

为了避免模糊、侧脸、遮挡、偏离画面中心等帧对平滑结果造成污染，系统为每个分析帧计算质量分：

```plain
q = 0.35*q_face
  + 0.20*q_area
  + 0.20*q_blur
  + 0.15*q_center
  + 0.10*q_bbox_stability
```

各项含义：

| 指标 | 含义 | 实现方式 |
|---|---|---|
| `q_face` | 人脸置信度 | DeepFace/extract_faces 返回的 confidence |
| `q_area` | 人脸大小是否合理 | bbox 面积 / 图像面积 |
| `q_blur` | 是否模糊 | Laplacian variance |
| `q_center` | 人脸是否偏离中心 | bbox 中心点与画面中心距离 |
| `q_bbox_stability` | 人脸框是否连续 | 当前 bbox 与上一 bbox 的 IoU |

最终，质量越低的帧在平滑中权重越小。

---

## 5. 功能三：视频连续性检测与平滑截断

如果视频中间出现断裂、切镜头、离屏重入、身份切换，不能把断裂前后的表情混在一起平滑。因此系统引入 `segment_id`。

触发新片段的条件包括：

```plain
1. 连续无脸超过阈值
2. 当前帧与上一帧画面差异过大
3. 两次分析之间时间间隔过大
4. bbox 跳变严重
5. 识别到的人物身份发生变化
```

一旦触发：

```plain
segment_id += 1
清空平滑缓存
后续平滑不跨 segment 进行
```

---

## 6. 功能四：人物建档与个体化表情适配

### 6.1 人脸建档

前端新增“个体化表情建档”区域，可以对特定人物建立本地档案：

```plain
输入人物名称
  → 创建档案
  → 采集当前人脸图像
  → DeepFace.represent 提取 Facenet512 embedding
  → 保存到 data/face_profiles/<person_id>/embeddings.npy
```

运行时，系统会提取当前人脸 embedding，与数据库中人物 embedding 做 cosine distance 匹配。

### 6.2 个体化表情样本

对某个人，可以手动标注表情样本：

```plain
选择人物
选择标签 happy / neutral / angry / sad / surprise / fear / disgust
点击“保存当前识别为标注样本”
```

样本保存为：

```plain
emotion_samples.csv
```

每条样本包含：

```plain
created_at, label, quality_score, happy, neutral, surprise, sad, angry, fear, disgust
```

### 6.3 连续采样 5 秒

新增“连续采样 5 秒”按钮，用于快速采集高质量样本：

```plain
选择人物
选择表情标签
设置采样时长，默认 5 秒
设置最低质量分，默认 0.55
点击“连续采样 5 秒”
```

系统会在指定时长内每隔约 450ms 尝试保存一条样本；如果当前帧质量分低于阈值或没有可用识别结果，则自动跳过。

推荐样本量：

```plain
至少 2 个表情类别
每个已采集类别至少 5 条
总样本数至少 20 条
```

### 6.4 个人适配器模型

个人适配器没有引入 PyTorch/sklearn，而是用 NumPy 实现了一个轻量 softmax 分类器：

```plain
输入 x = [7维 DeepFace 表情概率 + quality_score]
输出 y = 7类个人化表情概率
```

训练完成后保存为：

```plain
adapter.npz
```

推理时融合：

```plain
P_final = (1 - λ) * P_deepface + λ * P_personal_adapter
```

其中 `λ` 可在前端配置中调整。

### 6.5 新增验证指标

为了避免只看 train accuracy 导致过拟合，训练后新增：

```plain
train_accuracy
leave_one_out_accuracy
label_distribution
min_samples_per_present_label
validation_warnings
```

`leave_one_out_accuracy` 是小样本场景下更诚实的验证指标：每次留出 1 条样本做测试，其余样本训练，重复 N 次后计算准确率。

如果样本过少，前端会提示：

```plain
总样本数少于 20：个人适配器仅适合课堂演示，不建议强解释性能。
部分已采集类别少于 5 条：该类别容易过拟合。
少于 2 个表情类别：无法训练可靠的区分模型。
```

---

## 7. 功能五：实验结果自动导出

新增脚本：

```plain
export_experiment_report.py
```

它可以从 `analyze_video_offline.py` 生成的 JSON 中自动导出报告用实验结果，包括：

```plain
raw vs smooth 情绪跳变次数
no-face rate
平均质量分
segment 数量
平滑前后 dominant emotion 曲线图
```

输出文件：

```plain
summary.json        # 机器可读指标
summary.csv         # 可直接放入报告的单行表格
frame_records.csv   # 每帧 raw/smooth/quality 记录
emotion_curve.png   # raw vs smooth dominant emotion 曲线图
```

---

## 8. 如何运行实时系统

进入 demo 目录：

```bash
cd websocket-demo
```

启动服务：

```bash
python server.py
```

浏览器打开：

```plain
http://localhost:8000
```

注意事项：

```plain
1. 需要系统已安装 ffmpeg，并且命令行可直接运行 ffmpeg -version。
2. 浏览器需要允许摄像头权限。
3. 第一次运行 DeepFace 可能会自动下载模型权重，请保持网络可用。
```

---

## 9. 如何运行离线视频分析与实验导出

第一步：分析视频并生成 JSON。

```bash
python analyze_video_offline.py \
  --input demo.mp4 \
  --output outputs/demo_result.json \
  --every 5 \
  --window 5 \
  --sigma 2.0 \
  --detector_backend retinaface
```

第二步：导出实验报告文件。

```bash
python export_experiment_report.py \
  --input outputs/demo_result.json \
  --output-dir outputs/demo_report
```

导出后可在以下位置找到结果：

```plain
outputs/demo_report/summary.csv
outputs/demo_report/frame_records.csv
outputs/demo_report/emotion_curve.png
```

报告中可以这样描述：

```plain
本文对同一段视频分别统计原始 DeepFace 输出与高斯平滑后的 dominant emotion 跳变次数。结果显示，平滑后跳变次数明显下降，说明质量感知高斯平滑能够提升实时表情识别系统的输出稳定性。
```

---

## 10. 推荐实验设计

建议录制 5 段 20~30 秒视频：

```plain
1. 正脸正常光照
2. 微笑 / happy
3. 皱眉 / angry
4. 侧脸或低头
5. 离屏或遮挡
```

每段视频分别运行：

```bash
python analyze_video_offline.py --input xxx.mp4 --output outputs/xxx.json
python export_experiment_report.py --input outputs/xxx.json --output-dir outputs/report_xxx
```

最终整理表格：

| 场景 | raw 跳变次数 | smooth 跳变次数 | 降低比例 | no-face rate | 平均质量分 | segment 数 |
|---|---:|---:|---:|---:|---:|---:|
| 正脸 | | | | | | |
| 微笑 | | | | | | |
| 皱眉 | | | | | | |
| 侧脸 | | | | | | |
| 遮挡/离屏 | | | | | | |

---

## 11. 隐私与边界说明

`data/face_profiles/` 会保存人脸图片、embedding、个体化样本和适配器参数，属于生物特征相关数据。课程演示中建议：

```plain
1. 只采集自己或已明确同意的同学。
2. 不把 data/face_profiles/ 上传到公开仓库。
3. 报告中说明该系统用于课堂实验，不用于真实心理诊断。
4. 情绪水位只表示模型输出的交互指标，不代表真实心理状态。
```

---

## 12. 一句话总结

本次改造将原始 DeepFace 实时表情识别 Demo 从“单帧调包展示”升级为：

```plain
实时 DeepFace 表情识别
+ 质量感知高斯时序平滑
+ 视频连续性截断
+ 人脸建档与身份匹配
+ 个体化表情适配器
+ 自动实验结果导出
```

因此，它不仅能演示摄像头表情识别，还能用实验表格和曲线证明系统稳定性改进。
