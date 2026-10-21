# YOLOv8 + ArcFace 人脸识别与检索系统

端到端的人脸识别系统：**人脸检测 → 关键点对齐 → 512 维特征提取 → 底库 1:N 余弦相似度检索**。支持单图、批量目录、视频文件、摄像头实时四种输入，输出带标注的可视化结果和 CSV / JSON 检索报告。

## 功能

- **两种 1:N 检索实现**
  - `face_search.py`：InsightFace 单框架版，检测 + 识别一条链路。
  - `face_search_yolo.py`：YOLOv8 检测 + SCRFD 5 点关键点 + ArcFace 识别的双模型联合管线。
- **四种输入模式**：单图、批量目录、视频、摄像头实时。
- **底库生命周期**：批量构建、增量添加、按姓名删除、重名保护。
- **自动标注**：用 InsightFace 检测器批量生成 YOLO 格式标签，避免人工框脸。
- **数据集划分**：80/20 训练/验证，固定随机种子，可复现。
- **实时优化**：视频/摄像头跳帧检测、摄像头轻量模型、GPU / CPU 双端。
- **结构化输出**：标注图、标注视频、CSV / JSON 检索报告。

## 效果

在本机验证集上的检测指标：

| 指标 | 数值 |
|---|---|
| Precision | 88.7% |
| Recall | 93.1% |
| mAP50 | 97.3% |
| mAP50-95 | 92.0% |

- 数据集：5500 张人脸图，自动标注后按 4400 / 1100 划分训练集和验证集。
- 底库：2883 人 × 512 维 ArcFace 特征。

> 以上为本机验证结果，换数据分布会有差异。

## 处理流程

```
图片 / 视频 / 摄像头
        │
        ▼
YOLOv8s 单类人脸检测
        │
        ▼
SCRFD 5 点关键点对齐
        │
        ▼
ArcFace 512 维特征
        │
        ▼
底库余弦相似度 1:N 检索
        │
        ▼
阈值判定 → 标注结果 + CSV / JSON 报告
```

## 项目结构

| 文件 | 作用 |
|---|---|
| `face_search.py` | InsightFace 单框架版 1:N 检索，支持四种输入模式 |
| `face_search_yolo.py` | YOLOv8 + SCRFD + ArcFace 联合管线 |
| `detector.py` | YOLOv8 检测器封装 |
| `build_database.py` | 从图片目录批量构建 `.npz` 底库 |
| `add_to_database.py` | 增量添加人脸，支持重名保护 |
| `remove_from_database.py` | 按姓名删除底库条目 |
| `auto_label.py` | RetinaFace 自动标注，生成 YOLO `.txt` 标签 |
| `prepare_dataset.py` | 80/20 划分数据集并生成 `data.yaml` |
| `arcface_demo.py` | ArcFace 特征提取最小示例 |

`face_dataset/`、`faces/`、`runs/`、`search_results/` 是本机数据、训练输出和检索结果，不纳入版本库。

## 快速开始

### 1. 环境

- Python 3.10+
- Windows / Linux 均可；有 NVIDIA GPU 时速度更快，CPU 也能跑
- 安装依赖：

```bash
pip install ultralytics insightface onnxruntime opencv-python numpy tqdm
```

> Windows 上如果 `insightface` 安装失败，先安装 Visual C++ Build Tools，再重试。

### 2. 准备人脸图片

把图片放到一个目录，例如 `faces/faces/images/`。文件名（去掉扩展名）会作为底库里的姓名。

### 3. 自动标注并划分数据集

先改脚本顶部的路径配置：

- `auto_label.py`：`IMAGES_DIR`
- `prepare_dataset.py`：`SOURCE_DIR`、`DATASET_DIR`

```bash
python auto_label.py
python prepare_dataset.py
```

`prepare_dataset.py` 会生成 YOLO 需要的目录结构和 `face_dataset/data.yaml`。

### 4. 训练人脸检测器

```bash
yolo detect train data=face_dataset/data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=8
```

训练完成后，把 `best.pt` 的路径填到 `face_search_yolo.py` 顶部的 `YOLO_MODEL`，或者放到脚本默认位置。

### 5. 构建底库

改 `build_database.py` 顶部的 `IMAGE_DIR` 和 `OUTPUT_FILE`，然后：

```bash
python build_database.py
```

默认输出 `face_database.npz`。

### 6. 跑检索

```bash
# 单图（换成你自己的图片）
python face_search_yolo.py --mode image --input path/to/your_image.jpg --database face_database.npz --output-dir search_results

# 批量目录
python face_search_yolo.py --mode batch --input path/to/faces_dir --database face_database.npz

# 视频，每 3 帧检测 1 帧
python face_search_yolo.py --mode video --input path/to/video.mp4 --skip-frames 2 --database face_database.npz

# 摄像头实时
python face_search_yolo.py --mode camera --database face_database.npz
```

InsightFace 单框架版用法相同，把脚本换成 `face_search.py`。

常用参数：

| 参数 | 说明 |
|---|---|
| `--threshold` | 相似度阈值，默认 0.7，低于阈值标为 Unknown |
| `--database` | 底库 `.npz` 路径 |
| `--output-dir` | 结果输出目录 |
| `--gpu` | `-1` 用 CPU，`0` 用第一块 GPU |
| `--skip-frames` | 视频/摄像头跳帧数，越大越快但可能漏检 |
| `--yolo-conf` | YOLO 检测置信度阈值，默认 0.5 |

### 7. 维护底库

```bash
# 添加单张
python add_to_database.py --image xiaoming.jpg --name xiaoming --database face_database.npz

# 批量添加
python add_to_database.py --dir new_faces --database face_database.npz

# 删除
python remove_from_database.py --name xiaoming --database face_database.npz --dry-run
```

## 配置说明

部分脚本保留了作者本机的 Windows 路径（如 `C:\yolo_learning\...`）。换机器运行时，把下面这些配置改成你的路径：

| 文件 | 配置项 |
|---|---|
| `face_search.py` | `--database`、`--output-dir` |
| `face_search_yolo.py` | `YOLO_MODEL`、`DB_PATH`、`--database`、`--output-dir` |
| `build_database.py` | `IMAGE_DIR`、`OUTPUT_FILE`、`FaceAnalysis(root=...)` |
| `auto_label.py` | `IMAGES_DIR` |
| `prepare_dataset.py` | `SOURCE_DIR`、`DATASET_DIR` |
| `add_to_database.py` / `remove_from_database.py` | `DEFAULT_DB` |

> `build_database.py` 里的 InsightFace `root` 默认是作者本机的 `C:\Users\...\.insightface`。换机器时改成你的 `~/.insightface`，或删掉 `root` 参数让 InsightFace 用默认目录。

## 已知限制

- 1:N 检索目前是线性余弦相似度；底库到十万级时，建议换 FAISS / Milvus 等向量索引。
- 侧脸、遮挡、低光照、强逆光会明显掉点。
- 阈值 0.7 是经验值，需要按场景用验证集重新标定。
- 没有活体检测，不适合门禁、支付等高安全场景。
- 仅用于学习与研究。使用人脸图片前请确认你有权使用，不要把他人人脸数据或底库文件提交到公开仓库。

## 可能的扩展

- 向量索引：FAISS / Milvus，支持更大底库和更快检索。
- 活体检测：防止照片、视频翻拍攻击。
- Web 服务：把检索封装成 API，接前端或业务系统。
- 评测：补充不同光照、角度、遮挡条件下的分层指标。
