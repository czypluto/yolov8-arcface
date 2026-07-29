"""
自动标注脚本：用 InsightFace 的 RetinaFace 检测器对所有图片做人脸标注，
生成 YOLO 格式的 .txt 标签文件。

YOLO 标签格式（每行一个目标）:
  class_id x_center y_center width height
  坐标归一化到 0~1

输入: C:\yolo_learning\faces\faces\images\ 下的所有 .jpg 文件
输出: 每张图片同目录下生成同名 .txt 文件
"""

import os
import cv2
from insightface.app import FaceAnalysis
import numpy as np
from tqdm import tqdm
import time

# ============ 配置 ============
IMAGES_DIR = r"C:\yolo_learning\faces\faces\images"
CONF_THRESHOLD = 0.5   # 人脸检测置信度阈值，低于此值不标注
DET_SIZE = (640, 640)  # 检测分辨率，越大检出更多小脸但更慢
# ==============================

def main():
    # 1. 收集所有 jpg 文件
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"找到 {len(image_files)} 张图片")

    if not image_files:
        print("没有找到图片，请检查路径")
        return

    # 2. 加载 InsightFace（只用检测功能）
    print("加载 InsightFace 检测模型...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=DET_SIZE)  # ctx_id=-1 用 CPU
    print("模型加载完成\n")

    # 3. 逐张检测并生成标注
    stats = {
        'total': len(image_files),
        'annotated': 0,      # 至少检测到 1 张人脸的图片数
        'empty': 0,          # 未检测到人脸的图片数
        'total_faces': 0,     # 总人脸数
        'errors': 0,
    }

    start_time = time.time()

    for i, filename in enumerate(tqdm(image_files, desc="标注进度")):
        img_path = os.path.join(IMAGES_DIR, filename)
        label_path = os.path.join(IMAGES_DIR, filename.rsplit('.', 1)[0] + '.txt')

        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                stats['errors'] += 1
                continue

            h, w = img.shape[:2]

            # 检测人脸
            faces = app.get(img)

            # 过滤低置信度，写入 YOLO 格式
            lines = []
            for face in faces:
                if face.det_score >= CONF_THRESHOLD:
                    # bbox: [x1, y1, x2, y2] → YOLO 格式
                    x1, y1, x2, y2 = face.bbox
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # class_id = 0 (人脸)，坐标 clamp 到 0~1
                    xc = max(0.0, min(1.0, x_center))
                    yc = max(0.0, min(1.0, y_center))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    stats['total_faces'] += 1

            # 写入标签文件
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))

            if len(lines) > 0:
                stats['annotated'] += 1
            else:
                stats['empty'] += 1
                # 没检测到人脸时写入空文件（YOLO 训练要求图片和标签一一对应）
                with open(label_path, 'w') as f:
                    f.write('')

        except Exception as e:
            stats['errors'] += 1
            print(f"  错误: {filename} - {e}")

    elapsed = time.time() - start_time

    # 4. 打印统计
    print("\n" + "=" * 50)
    print("自动标注完成!")
    print("=" * 50)
    print(f"  总图片数:     {stats['total']}")
    print(f"  有标注的图片: {stats['annotated']}")
    print(f"  无人脸的图片: {stats['empty']}")
    print(f"  总人脸数:     {stats['total_faces']}")
    print(f"  错误数:       {stats['errors']}")
    print(f"  平均每张人脸: {stats['total_faces']/max(stats['annotated'],1):.2f}")
    print(f"  耗时:         {elapsed:.1f} 秒")
    print("=" * 50)

if __name__ == '__main__':
    main()
