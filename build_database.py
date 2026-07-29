import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ========== 配置 ==========
IMAGE_DIR = r"C:\yolo_learning\faces\faces\images"  # 你的图片目录
OUTPUT_FILE = r"C:\yolo_learning\face_database.npz"  # 底库保存路径
DET_SIZE = (640, 640)
# ==========================

# 1. 加载 ArcFace 模型
app = FaceAnalysis(name='buffalo_l', root=r'C:\Users\24522\.insightface')
app.prepare(ctx_id=-1, det_size=DET_SIZE)

# 2. 收集所有图片
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"找到 {len(image_files)} 张图片")

# 3. 逐张提取特征
database = {}  # {文件名(作为人名): 特征向量}

for filename in image_files:
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"  跳过: {filename} (读取失败)")
        continue

    # 检测人脸并提取特征
    faces = app.get(img)

    if not faces:
        print(f"  跳过: {filename} (未检测到人脸)")
        continue

    # 取置信度最高的人脸
    face = max(faces, key=lambda f: f.det_score)

    # 用人名 = 文件名（去掉扩展名）
    name = filename.rsplit('.', 1)[0]
    database[name] = face.embedding

    print(f"  ✓ {name}: 特征维度 {face.embedding.shape}")

print(f"\n底库构建完成: {len(database)} 人")

# 4. 保存底库
np.savez(OUTPUT_FILE,
         names=np.array(list(database.keys())),
         embeddings=np.array(list(database.values())))

print(f"底库已保存到: {OUTPUT_FILE}")
print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")