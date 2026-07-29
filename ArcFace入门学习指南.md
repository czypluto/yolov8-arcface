# ArcFace 入门学习指南（RTX 3050 Ti 6GB 版）

## 一、ArcFace 是什么

ArcFace（Additive Angular Margin Loss）是一种**人脸识别**算法，由中科院/旷视科技在 2019 年提出。它的核心贡献不是网络结构，而是损失函数——通过在角度空间引入惩罚项，让**同一人的特征靠得更近、不同人的特征离得更远**。

> 对比 YOLO：YOLO 解决"目标在哪、是什么类别"，ArcFace 解决"这张脸是谁的"。

**在工业界，你不需要从头实现 ArcFace 论文的公式**。开源库 InsightFace 已经把它封装好了，你只需要调 API。

---

## 二、安装（10 分钟搞定）

### 2.1 激活虚拟环境

```cmd
cd C:\yolo_learning
venv\Scripts\activate
```

### 2.2 安装 InsightFace

```cmd
pip install insightface onnxruntime-gpu
```

> `onnxruntime-gpu` 让推理跑在 GPU 上（CPU 版用 `onnxruntime`）。你的 3050 Ti 用 GPU 版。

### 2.3 下载预训练模型

InsightFace 提供几个打包好的模型组合：

| 模型包 | 大小 | 包含功能 | 推荐场景 |
|--------|------|---------|---------|
| **buffalo_l** | ~350MB | 检测 + 关键点 + 识别 | ✅ **推荐入门**，一站式搞定 |
| **buffalo_s** | ~90MB | 检测 + 关键点 + 识别（轻量） | 显存紧张时用 |
| **antelopev2** | ~450MB | 检测 + 关键点 + 识别（最强） | 精度优先 |

```python
from insightface.app import FaceAnalysis

# 首次运行会自动下载模型到 ~/.insightface/models/
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
print("模型加载成功！")
```

### 2.4 验证安装

```cmd
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider','CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(320,320)); print('ArcFace 安装成功!')"
```

> 首次运行会下载约 350MB 模型文件，耐心等待。

---

## 三、核心概念：ArcFace 到底在做什么

### 3.1 人脸识别三步走

```
输入图片
   ↓
① 人脸检测（Face Detection）     →  找到人脸框 (x1,y1,x2,y2)
   ↓
② 关键点定位（Landmark）          →  找到 5 个关键点（双眼、鼻尖、嘴角）
   ↓
③ 人脸对齐 + 特征提取（ArcFace）  →  输出 512 维特征向量
   ↓
④ 特征比对（Cosine Similarity）   →  与底库比对，输出是谁
```

### 3.2 特征向量是什么

ArcFace 把一张人脸压缩成一个 **512 维的浮点数向量**（类似于把人脸变成一串数字指纹）。特征向量之间的**余弦相似度**越大，说明越可能是同一个人。

```
张三的特征: [0.12, -0.34, 0.56, ..., 0.78]  ← 512 个数
李四的特征: [0.89, 0.23, -0.67, ..., 0.11]  ← 512 个数

余弦相似度 = 0.35  → 不是同一个人
余弦相似度 = 0.92  → 很可能是同一个人
```

### 3.3 判断阈值

| 余弦相似度 | 含义 |
|-----------|------|
| > 0.5 | 可能是同一人 |
| > 0.6 | 较有可能是同一人 |
| **> 0.7** | **工业推荐阈值** |
| > 0.8 | 非常确定 |

---

## 四、第一步：跑通 Demo

### 4.1 单张图片提取人脸特征

```python
import cv2
from insightface.app import FaceAnalysis

# 加载模型（GPU 加速）
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 读取图片
img = cv2.imread('bus.jpg')
faces = app.get(img)

print(f"检测到 {len(faces)} 张人脸")

for i, face in enumerate(faces):
    print(f"\n--- 人脸 {i+1} ---")
    print(f"  边界框: {face.bbox}")           # [x1, y1, x2, y2]
    print(f"  置信度: {face.det_score:.4f}")   # 检测置信度
    print(f"  关键点: {face.kps}")             # 5 个关键点坐标
    print(f"  特征维度: {face.embedding.shape}")  # (512,)
    print(f"  特征前5维: {face.embedding[:5]}")
```

运行：

```cmd
python arcface_demo.py
```

### 4.2 两个人脸比对（判断是不是同一个人）

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 读取两张图片
img1 = cv2.imread('person_a.jpg')
img2 = cv2.imread('person_b.jpg')

faces1 = app.get(img1)
faces2 = app.get(img2)

if faces1 and faces2:
    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    # 计算余弦相似度
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"相似度: {similarity:.4f}")
    print(f"判定: {'同一人' if similarity > 0.7 else '不同人'}")
else:
    print("某张图片未检测到人脸")
```

---

## 五、第二步：构建人脸底库 + 识别

这是 ArcFace 最核心的实战场景：**先注册，后识别**。

### 5.1 完整人脸识别系统

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.database = {}  # {姓名: 特征向量}

    def register(self, image_path, name):
        """注册人脸到底库"""
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if not faces:
            print(f"❌ {image_path} 未检测到人脸")
            return False

        # 取置信度最高的人脸
        face = max(faces, key=lambda f: f.det_score)
        self.database[name] = face.embedding
        print(f"✅ 注册成功: {name}, 特征维度: {face.embedding.shape}")
        return True

    def recognize(self, image_path, threshold=0.6):
        """识别人脸"""
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        if not faces:
            print("❌ 未检测到人脸")
            return None

        results = []
        for face in faces:
            best_match = None
            best_sim = -1

            for name, emb_db in self.database.items():
                sim = np.dot(face.embedding, emb_db) / (
                    np.linalg.norm(face.embedding) * np.linalg.norm(emb_db)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_match = name

            if best_sim >= threshold:
                results.append({
                    'name': best_match,
                    'similarity': best_sim,
                    'bbox': face.bbox.astype(int)
                })
            else:
                results.append({
                    'name': '未知',
                    'similarity': best_sim,
                    'bbox': face.bbox.astype(int)
                })

        return results

# --- 使用示例 ---
if __name__ == '__main__':
    fr = FaceRecognizer()

    # 1. 注册底库（用每人 1~3 张正面照）
    fr.register('zhangsan_1.jpg', '张三')
    fr.register('lisi_1.jpg', '李四')
    fr.register('wangwu_1.jpg', '王五')

    # 2. 识别
    results = fr.recognize('test_photo.jpg')
    for r in results:
        print(f"  {r['name']} (相似度: {r['similarity']:.4f}) 在 {r['bbox']}")
```

### 5.2 底库管理最佳实践

| 要点 | 建议 |
|------|------|
| 每人注册照 | 1~3 张正面、光线好的照片 |
| 存储格式 | 用 `numpy.save()` 或 pickle 序列化特征向量 |
| 特征向量大小 | 每人约 2KB（512 float），10 万人约 200MB |
| 底库更新 | 增量添加，不需要重新计算所有人 |

---

## 六、第三步：YOLOv8s + ArcFace 联合部署

这是你之前说的搭配方案。YOLOv8s 负责"找人"，ArcFace 负责"认人"。

```python
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

class YoloArcfaceSystem:
    def __init__(self):
        # YOLO 检测行人
        self.detector = YOLO('yolov8s.pt')

        # ArcFace 识别人脸
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 人脸底库
        self.database = {}

    def register(self, image_path, name):
        """注册人脸"""
        img = cv2.imread(image_path)
        faces = self.face_app.get(img)
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            self.database[name] = face.embedding
            print(f"注册: {name}")

    def detect_and_recognize(self, image_path, threshold=0.6):
        """检测行人 → 裁剪 → 人脸识别"""
        img = cv2.imread(image_path)

        # 1. YOLO 检测行人（class 0 = person）
        results = self.detector(image_path)
        persons = results[0].boxes[results[0].boxes.cls == 0]

        detections = []
        for box in persons:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 2. 从行人区域提取人脸
            person_crop = img[y1:y2, x1:x2]
            faces = self.face_app.get(person_crop)

            person_info = {
                'bbox': [x1, y1, x2, y2],
                'faces': []
            }

            # 3. ArcFace 识别每张人脸
            for face in faces:
                best_name = '未知'
                best_sim = -1
                for name, emb in self.database.items():
                    sim = np.dot(face.embedding, emb) / (
                        np.linalg.norm(face.embedding) * np.linalg.norm(emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                if best_sim >= threshold:
                    best_name = best_name
                else:
                    best_name = '未知'

                person_info['faces'].append({
                    'name': best_name,
                    'similarity': best_sim,
                    'face_bbox': face.bbox.astype(int).tolist()
                })

            detections.append(person_info)

        return detections

# --- 使用示例 ---
if __name__ == '__main__':
    system = YoloArcfaceSystem()

    # 注册
    system.register('zhangsan.jpg', '张三')
    system.register('lisi.jpg', '李四')

    # 检测 + 识别
    results = system.detect_and_recognize('classroom.jpg')
    for det in results:
        for face in det['faces']:
            print(f"  {face['name']} (相似度: {face['similarity']:.4f}) "
                  f"在行人区域 {det['bbox']}")
```

---

## 七、第四步：进阶功能

### 7.1 实时摄像头识别

```python
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

database = {}  # 先注册，参考 5.1 节
THRESHOLD = 0.6

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        # 比对底库
        best_name, best_sim = '未知', 0
        for name, emb in database.items():
            sim = np.dot(face.embedding, emb) / (
                np.linalg.norm(face.embedding) * np.linalg.norm(emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name if sim >= THRESHOLD else '未知'

        # 画框 + 写名字
        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 255, 0) if best_sim >= THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{best_name} {best_sim:.2f}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 7.2 批量提取特征向量（构建底库）

```python
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 批量扫描文件夹（每个子文件夹 = 一个人）
faces_dir = 'faces_database'
embeddings = {}  # {姓名: [特征1, 特征2, ...]}

for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings[person_name] = []
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(person_dir, img_file))
        faces = app.get(img)
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            embeddings[person_name].append(face.embedding)

    print(f"{person_name}: {len(embeddings[person_name])} 张有效人脸")

# 对每人取平均特征（提高鲁棒性）
avg_embeddings = {}
for name, embs in embeddings.items():
    avg_embeddings[name] = np.mean(embs, axis=0)
    print(f"{name} 平均特征维度: {avg_embeddings[name].shape}")

# 保存底库
np.savez('face_database.npz', names=list(avg_embeddings.keys()),
         embeddings=np.array(list(avg_embeddings.values())))
print("底库已保存到 face_database.npz")
```

### 7.3 Faiss 加速大规模检索（1 万+ 人脸底库）

```cmd
pip install faiss-gpu
```

```python
import faiss
import numpy as np

# 假设数据库有 N 个人
names = ['张三', '李四', '王五', ...]        # N 个名字
vectors = [...]                               # N x 512 特征矩阵

# 构建 FAISS 索引（GPU 加速）
dim = 512
index = faiss.IndexFlatIP(dim)  # 内积索引（向量已归一化则等价于余弦）
index.add(np.array(vectors).astype('float32'))

# 查询：输入一张人脸特征，返回最相似的 Top-K
query = face.embedding.reshape(1, -1).astype('float32')
scores, indices = index.search(query, k=5)

for i in range(len(indices[0])):
    idx = indices[0][i]
    if idx >= 0:
        print(f"  {i+1}. {names[idx]} (相似度: {scores[0][i]:.4f})")
```

> FAISS 能让 10 万人脸底库的检索时间从暴力遍历的秒级降到毫秒级。

---

## 八、常见问题

### Q1: 3050 Ti 显存不够？

```python
# 减小检测分辨率（牺牲精度换速度）
app.prepare(ctx_id=0, det_size=(320, 320))  # 默认 640

# 如果 GPU 还是不够，回落到 CPU
app.prepare(ctx_id=-1, det_size=(640, 640))
```

### Q2: 识别准确率低？

- 确保注册照**正面、光线好、无遮挡**
- 每人注册 **2~3 张**不同角度的照片，取平均特征
- 调高/调低阈值：安全场景用 0.7，便利场景用 0.5
- 测试图片和注册照**拍摄条件差距太大**（如口罩、墨镜）会大幅降准

### Q3: onnxruntime-gpu 安装失败？

```cmd
# 如果 GPU 版安装有问题，先用 CPU 版（慢一点但能跑）
pip install onnxruntime

# CPU 版使用方式
app.prepare(ctx_id=-1, det_size=(640, 640))
```

### Q4: CUDA 和 onnxruntime 的 CUDA 版冲突？

InsightFace 用 ONNX Runtime 而不是 PyTorch，所以 CUDA 版本**不需要和 PyTorch 一致**。`onnxruntime-gpu` 自带 CUDA runtime。

### Q5: 和 YOLOv8s 搭配时，谁负责检测人脸？

两种选择：

| 方案 | 检测器 | 优点 | 缺点 |
|------|--------|------|------|
| **A（推荐）** | InsightFace 内置的 RetinaFace | 专门做人脸检测，精度高 | 额外加载一个模型 |
| **B** | YOLOv8s 检测 person → ArcFace 识别人脸 | 只需一个检测模型 | YOLOv8s 不是专门人脸检测，小脸可能漏检 |

> 如果只需要人脸识别（不需要先检测行人再找脸），直接用方案 A，不需要 YOLOv8s。

---

## 九、学习路径建议

| 阶段 | 内容 | 预期时间 |
|------|------|---------|
| **第1天** | 安装 + 跑通 Demo + 理解特征向量 | 1 小时 |
| **第2天** | 构建人脸底库 + 识别系统 | 2 小时 |
| **第3天** | 实时摄像头识别 | 1 小时 |
| **第4天** | YOLOv8s + ArcFace 联合部署 | 2 小时 |
| **第5天** | 优化：Faiss 加速 + 多角度注册 | 2 小时 |
| **第6~7天** | 做小项目（考勤系统/门禁模拟） | 3 小时 |

---

## 十、小项目练手推荐

| 项目 | 难度 | 说明 |
|------|------|------|
| **人脸相册管理** | ⭐ | 注册 10 个同学 → 自动分类照片 |
| **考勤打卡系统** | ⭐⭐ | 摄像头实时识别 + 记录到 CSV |
| **门禁模拟** | ⭐⭐ | 摄像头识别 → 匹配底库 → 显示通过/拒绝 |
| **视频人脸追踪** | ⭐⭐⭐ | 视频中持续追踪并识别同一个人 |

---

## 十一、完整命令速查

```cmd
:: 安装
pip install insightface onnxruntime-gpu

:: 验证
python -c "from insightface.app import FaceAnalysis; print('安装成功')"

:: 如果 GPU 版失败，回落 CPU
pip install onnxruntime
```

---

## 十二、推荐学习资源

| 资源 | 链接 |
|------|------|
| InsightFace 官方文档 | https://insightface.ai |
| InsightFace GitHub | https://github.com/deepinsight/insightface |
| ArcFace 论文 | https://arxiv.org/abs/1801.07698 |
| ONNX Runtime 文档 | https://onnxruntime.ai |
| Faiss 文档 | https://faiss.ai |

  # 1. 单张图片查询

  python -X utf8 face_search.py --mode image --input bus.jpg

  # 2. 批量图片查询

  python -X utf8 face_search.py --mode batch --input C:\yolo_learning\faces\faces\images

  # 3. 摄像头实时比对

  python -X utf8 face_search.py --mode camera

  # 4. 视频文件比对

  python -X utf8 face_search.py --mode video --input video.mp4 --skip-frames 5

 # 纯 InsightFace 版 — 通用人脸检测，适合任意图片
  python -X utf8 face_search.py --mode camera

  # YOLO + ArcFace 版 — 你训练的人脸检测器 + ArcFace 识别
  python -X utf8 face_search_yolo.py --mode camera

python -X utf8 face_search_yolo.py --mode image --input C:\yolo_learning\faces\faces\images\ftw1.jpg

  # C:\yolo_learning\add_to_database.py用法.

  # 添加单张图片，人名 = 文件名（如 xiaoming.jpg → "xiaoming"）

  python add_to_database.py --image C:\photos\xiaoming.jpg

  # 自定义人名
  python add_to_database.py --image C:\photos\face.jpg --name zhangsan

  # 强制覆盖已有的同名记录
  python add_to_database.py --image zhangsan.jpg --force

  # 批量添加整个目录
  python add_to_database.py --dir C:\photos\new_faces

  # 指定其他底库文件
  python add_to_database.py --image face.jpg --database my_custom_db.npz

 # C:\yolo_learning\remove_from_database.py用法.

 # 删除一个人

python remove_from_database.py --name xiaoming

  # 一次删多个
  python remove_from_database.py --name zhangsan --name lisi

  # 先看匹配结果，确认再动手（安全，不实际删除）
  python remove_from_database.py --name xiaoming --dry-ru
