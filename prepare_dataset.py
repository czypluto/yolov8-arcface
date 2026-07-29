"""
整理 YOLO 训练数据集：
1. 把 5500 张图片+标签按 80/20 划分 train/val
2. 创建 YOLO 要求的目录结构
3. 生成 data.yaml

YOLO 要求的目录结构:
  dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── data.yaml
"""

import os
import shutil
import random
from tqdm import tqdm

# ============ 配置 ============
SOURCE_DIR = r"C:\yolo_learning\faces\faces\images"  # 原图+标签目录
DATASET_DIR = r"C:\yolo_learning\face_dataset"         # 输出数据集目录
TRAIN_RATIO = 0.8                                       # 训练集比例
SEED = 42                                               # 随机种子（保证可复现）
# ==============================

def main():
    random.seed(SEED)

    # 1. 收集所有配对的图片和标签
    image_files = sorted([
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # 只保留有对应 .txt 标签的图片
    pairs = []
    for img_file in image_files:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(SOURCE_DIR, label_file)
        if os.path.exists(label_path):
            pairs.append((img_file, label_file))

    print(f"找到 {len(pairs)} 对 图片+标签")

    # 2. 随机打乱并划分
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"训练集: {len(train_pairs)} 张")
    print(f"验证集: {len(val_pairs)} 张")

    # 3. 创建目录结构
    dirs = [
        os.path.join(DATASET_DIR, 'images', 'train'),
        os.path.join(DATASET_DIR, 'images', 'val'),
        os.path.join(DATASET_DIR, 'labels', 'train'),
        os.path.join(DATASET_DIR, 'labels', 'val'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 4. 复制文件（用符号链接/硬链接会更快，但 Windows 兼容性一般，直接复制）
    print("\n复制训练集...")
    for img_file, label_file in tqdm(train_pairs, desc="train"):
        shutil.copy2(
            os.path.join(SOURCE_DIR, img_file),
            os.path.join(DATASET_DIR, 'images', 'train', img_file)
        )
        shutil.copy2(
            os.path.join(SOURCE_DIR, label_file),
            os.path.join(DATASET_DIR, 'labels', 'train', label_file)
        )

    print("复制验证集...")
    for img_file, label_file in tqdm(val_pairs, desc="val"):
        shutil.copy2(
            os.path.join(SOURCE_DIR, img_file),
            os.path.join(DATASET_DIR, 'images', 'val', img_file)
        )
        shutil.copy2(
            os.path.join(SOURCE_DIR, label_file),
            os.path.join(DATASET_DIR, 'labels', 'val', label_file)
        )

    # 5. 生成 data.yaml
    yaml_content = f"""# YOLO 人脸检测数据集
# 自动生成

train: {DATASET_DIR}/images/train
val: {DATASET_DIR}/images/val

nc: 1
names: ['face']
"""

    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\ndata.yaml 已保存到: {yaml_path}")
    print("\n最终目录结构:")
    print(f"  {DATASET_DIR}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  ({len(train_pairs)} 张)")
    print(f"  │   └── val/    ({len(val_pairs)} 张)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/  ({len(train_pairs)} 个)")
    print(f"  │   └── val/    ({len(val_pairs)} 个)")
    print(f"  └── data.yaml")
    print("\n数据集整理完成!")

if __name__ == '__main__':
    main()
