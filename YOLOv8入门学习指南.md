# YOLOv8 入门学习指南（RTX 3050 Ti 6GB 版）

## 一、YOLO 是什么

YOLO = You Only Look Once（只看一眼），是目前最流行的**实时目标检测**框架。它能在图片或视频中**同时**找出所有目标的位置和类别，速度快到可以实时处理视频。

> 对比：传统方法需要"先扫描区域 → 再识别内容"两步，YOLO 一步搞定。

**YOLOv8** 是 Ultralytics 于 2023 年发布的版本，采用 Anchor-Free（无锚框）设计，相比 YOLOv5 精度更高、速度更快，并且检测、分割、分类、姿态估计四大任务共用同一套 API，是目前工业界用得最广泛的版本。

> 你本机已装好 `ultralytics 8.4.102`，并下载了 `yolov8n.pt`、`yolov8s.pt` 两个模型，可以直接开跑。

---

## 二、安装（已完成，这里供回顾）

### 2.1 创建虚拟环境（已完成）

```cmd
mkdir C:\yolo_learning
cd C:\yolo_learning
python -m venv venv
venv\Scripts\activate
```

> 你已经创建好了 `c:\yolo_learning\venv`，以后每次操作前先 `venv\Scripts\activate` 激活即可。

### 2.2 安装 Ultralytics（已完成）

```cmd
pip install ultralytics
```

Ultralytics 会自动帮你装好 PyTorch 和 CUDA 支持。你的环境里已包含 `torch`、`opencv-python`、`numpy`、`matplotlib` 等依赖。

### 2.3 验证安装

```cmd
python -c "from ultralytics import YOLO; import ultralytics; print('ultralytics', ultralytics.__version__); print('安装成功!')"
```

应输出 `ultralytics 8.4.102` 和 `安装成功!`。

### 2.4 检查 GPU 是否可用（3050 Ti）

```cmd
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> 显示 `CUDA 可用: True` 才能用 GPU 加速；若为 `False`，推理训练会回落到 CPU（慢很多，但也能跑）。

---

## 三、第一步：跑通检测 Demo（最直观的入门）

### 3.1 检测单张图片（用你已下载的模型）

```cmd
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

这条命令会自动：
1. 使用本地 `yolov8n.pt`（已下载，不再重复下载）
2. 下载测试图片 `bus.jpg`
3. 运行检测，输出结果到 `runs/detect/predict/`

**用你自己的图片**（你目录下已有 `h.png`）：

```cmd
yolo predict model=yolov8n.pt source=C:\yolo_learning\langdog.jpg
```

**换更大的模型对比精度**（你也有 `yolov8s.pt`）：

```cmd
yolo predict model=yolov8s.pt source=C:\yolo_learning\che.mp4
```

### 3.2 检测视频

```cmd
yolo predict model=yolov8n.pt source=C:\path\to\your_video.mp4
```

### 3.3 打开摄像头实时检测

```cmd
yolo predict model=yolov8n.pt source=0
```

> `source=0` 表示默认摄像头。按 `q` 退出。

### 3.4 n 和 s 的区别直观对比

同一张 `h.png`，分别用两个模型检测，观察：
- `yolov8n`：速度极快，但可能漏掉小目标或低置信度目标
- `yolov8s`：稍慢一点点，但检测更全、置信度更高

这就是"精度 vs 速度"的权衡，后续选型全靠这个直觉。

---

## 四、第二步：用 Python 代码调用 YOLO

创建 `demo.py`（建议放在 `c:\yolo_learning` 下）：

```python
from ultralytics import YOLO

# 加载模型（用你已下载的本地权重）
model = YOLO("yolov8n.pt")

# 检测图片（本地图片）
results = model("h.png")

# 显示结果
for r in results:
    print(f"检测到 {len(r.boxes)} 个目标")
    for box in r.boxes:
        cls = int(box.cls[0])          # 类别索引
        conf = float(box.conf[0])       # 置信度
        x1, y1, x2, y2 = box.xyxy[0]   # 边界框坐标
        print(f"  {model.names[cls]}: {conf:.2f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# 保存标注后的图片
results[0].save("output.jpg")
```

运行：

```cmd
python demo.py
```

**进阶：一次性对比 n 和 s 两个模型**

```python
from ultralytics import YOLO

for name in ["yolov8n.pt", "yolov8s.pt"]:
    model = YOLO(name)
    results = model("h.png")
    n = sum(len(r.boxes) for r in results)
    print(f"{name}: 检测到 {n} 个目标")
```

---

## 五、第三步：理解模型家族

YOLOv8 有 5 种规格，按体积从大到小：

| 模型 | 体积 | 3050 Ti 推理速度 | 精度 | 适合场景 |
|------|------|-----------------|------|---------|
| **yolov8n** | ~6MB | **>1000 FPS** | 低 | 手机/边缘设备 |
| **yolov8s** | ~22MB | **~800 FPS** | 中 | 实时视频（你已有） |
| **yolov8m** | ~52MB | **~500 FPS** | 中高 | 均衡选择 |
| **yolov8l** | ~87MB | **~300 FPS** | 高 | 高精度需求 |
| **yolov8x** | ~131MB | **~150 FPS** | 最高 | 离线分析 |

```python
# 切换模型很简单，改文件名就行
model = YOLO("yolov8n.pt")  # 最轻量（你已有）
model = YOLO("yolov8s.pt")  # 推荐日常用（你已有）
model = YOLO("yolov8m.pt")  # 精度优先（需要时再下）
```

> 你目前手上 n + s 两档，正好覆盖"最快"和"日常推荐"两个最常用场景，足够入门 + 做小项目。

---

## 六、第四步：自定义数据集标注

### 6.1 安装标注工具

**推荐：LabelImg（最简单）**

```cmd
pip install labelImg
labelImg
```

**备选：LabelStudio（功能更强）**

```cmd
pip install label-studio
label-studio start
```

### 6.2 标注流程

1. 收集你的图片（至少 100~200 张）
2. 用 LabelImg 打开图片文件夹
3. 框选目标 → 输入类别名称 → 保存
4. YOLO 格式的标注文件是 `.txt`，每行一个目标：
   ```
   class_id x_center y_center width height
   ```
   > 坐标是归一化的（0~1 之间）

### 6.3 标注文件示例

```
0 0.5 0.5 0.3 0.4    # 类别0，中心在(0.5,0.5)，宽0.3，高0.4
1 0.2 0.3 0.1 0.15   # 类别1，中心在(0.2,0.3)，宽0.1，高0.15
```

### 6.4 目录结构

```
C:\yolo_learning\dataset\
├── images\
│   ├── train\    # 训练图片（~80%）
│   └── val\      # 验证图片（~20%）
├── labels\
│   ├── train\    # 训练标注（与图片同名.txt）
│   └── val\      # 验证标注
└── data.yaml      # 数据集配置文件
```

### 6.5 创建 data.yaml

```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 2                       # 类别数量
names: ['cat', 'dog']       # 类别名称列表
```

---

## 七、第五步：自定义训练

### 7.1 训练命令

```cmd
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=100 batch=8 imgsz=640
```

> 从你已经下载的 `yolov8s.pt` 开始微调，比从头训练快很多、也准很多。

### 7.2 参数说明

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `model` | `yolov8s.pt` | 预训练模型，从它开始微调更快（你已有） |
| `data` | `data.yaml` | 你的数据集配置 |
| `epochs` | 100~300 | 训练轮数，越多越准 |
| `batch` | 8 | 3050 Ti 6GB 推荐 8，显存不够就 4 |
| `imgsz` | 640 | 输入图片大小，越大越准越慢 |
| `patience` | 50 | 连续 50 轮没提升就自动停止 |

### 7.3 训练时间预估（3050 Ti）

| 图片数 | 模型 | epochs | 时间 |
|--------|------|--------|------|
| 200 张 | yolov8s | 100 | ~30 分钟 |
| 500 张 | yolov8s | 100 | ~1 小时 |
| 1000 张 | yolov8s | 100 | ~2 小时 |
| 1000 张 | yolov8m | 100 | ~3 小时 |

### 7.4 训练过程监控

训练过程中会实时显示：
- `train/box_loss`：边界框损失（越低越好）
- `train/cls_loss`：分类损失（越低越好）
- `metrics/mAP50`：平均精度（越高越好，>0.5 算不错）

训练结果保存在 `runs/detect/train/`。

### 7.5 用训练好的模型检测

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 检测
results = model("test.jpg")
results[0].show()  # 显示结果
```

---

## 八、第六步：进阶功能

### 8.1 目标跟踪

```cmd
yolo track model=yolov8n.pt source=0  # 摄像头实时跟踪
```

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.track("video.mp4", persist=True)
for r in results:
    if r.boxes.id is not None:
        print(f"跟踪到 {len(r.boxes.id)} 个目标")
```

### 8.2 导出模型

```cmd
:: 导出 ONNX（跨平台部署）
yolo export model=yolov8s.pt format=onnx

:: 导出 TFLite（移动端）
yolo export model=yolov8s.pt format=tflite
```

### 8.3 分割（Instance Segmentation）

YOLOv8 的分割模型后缀是 `-seg`：

```cmd
:: 需要先下载 yolov8n-seg.pt（约 6MB）
yolo predict model=yolov8n-seg.pt source=h.png
```

### 8.4 姿态估计（Pose Estimation）

YOLOv8 的姿态模型后缀是 `-pose`：

```cmd
:: 需要先下载 yolov8n-pose.pt（约 7MB）
yolo predict model=yolov8n-pose.pt source=h.png
```

### 8.5 图像分类（Classification）

YOLOv8 的分类模型后缀是 `-cls`：

```cmd
yolo classify predict model=yolov8n-cls.pt source=h.png
```

---

## 九、学习路径建议

| 阶段 | 内容 | 预期时间 |
|------|------|---------|
| **第1天** | 跑通 Demo + 理解代码（你已具备条件） | 1 小时 |
| **第2天** | 收集图片 + 标注 100 张 | 2~3 小时 |
| **第3天** | 训练第一个自定义模型 + 调参 | 1 小时 |
| **第4~5天** | 优化数据集 + 提高精度 | 2 小时 |
| **第6天** | 做一个小项目（简易车牌检测/猫狗识别） | 2 小时 |
| **第7天** | 进阶：跟踪 + 部署 + 实时视频 | 2 小时 |

---

## 十、小项目练手推荐

| 项目 | 难度 | 数据集获取 |
|------|------|-----------|
| **口罩检测** | ⭐ | 自己拍 100 张，有/无口罩两类 |
| **车牌检测** | ⭐⭐ | 网上找停车场图片标注 |
| **猫咪品种识别** | ⭐⭐ | Kaggle 有现成数据集 |
| **交通标志检测** | ⭐⭐⭐ | 开源数据集 TT100K |
| **PCB 缺陷检测** | ⭐⭐⭐ | 开源 PCB 数据集 |

---

## 十一、常见问题

### Q1: 3050 Ti 显存不够怎么办？

```cmd
:: 减小 batch_size
yolo train model=yolov8n.pt batch=4

:: 用更小的输入尺寸
yolo train model=yolov8n.pt imgsz=416

:: 用 nano 模型
yolo train model=yolov8n.pt
```

### Q2: 模型精度不够？

- 增加训练数据量（至少 200 张/类）
- 用更大的模型（n → s → m → l，你已有 n 和 s）
- 增加 epochs（100 → 300）
- 检查标注质量（边界框是否准确）

### Q3: 训练时 loss 不下降？

- 检查 data.yaml 路径是否正确
- 检查标注文件是否为空
- 检查类别索引是否从 0 开始

### Q4: 检测结果不准？

- 先用预训练模型检测，看是不是数据问题
- 查看验证集标注是否准确
- 检查测试图片是否和训练数据分布一致

### Q5: yolov8n 和 yolov8s 怎么选？

- **追求速度**（实时视频、边缘部署）→ `yolov8n.pt`
- **追求精度**（离线分析、demo 展示）→ `yolov8s.pt`
- **入门学习**两个都试一遍，直观感受差异

---

## 十二、完整命令速查

```cmd
:: 安装
pip install ultralytics

:: 推理
yolo predict model=yolov8n.pt source=h.png
yolo predict model=yolov8s.pt source=h.png
yolo predict model=yolov8n.pt source=0

:: 训练
yolo train model=yolov8s.pt data=dataset/data.yaml epochs=100 batch=8 imgsz=640

:: 跟踪
yolo track model=yolov8n.pt source=video.mp4

:: 导出
yolo export model=yolov8s.pt format=onnx

:: 验证
yolo val model=runs/detect/train/weights/best.pt data=dataset/data.yaml
```

---

## 十三、推荐学习资源

| 资源 | 链接 |
|------|------|
| Ultralytics 官方文档 | https://docs.ultralytics.com |
| YOLOv8 官方仓库 | https://github.com/ultralytics/ultralytics |
| YOLO 论文 | https://arxiv.org/abs/1506.02640 |
| Roboflow 数据集平台 | https://roboflow.com |
| Kaggle 竞赛数据集 | https://kaggle.com |
