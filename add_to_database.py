"""
增量添加人脸图片到底库 — 无需重建整个底库

用法:
  # 添加单张图片，人名 = xiaoming
  python add_to_database.py --image C:\photos\xiaoming.jpg --name xiaoming

  # 添加单张图片，人名 = 文件名（去掉扩展名）
  python add_to_database.py --image C:\photos\xiaoming.jpg

  # 批量添加目录下所有图片，人名 = 各自文件名
  python add_to_database.py --dir C:\photos\new_faces

  # 强制覆盖重名的人
  python add_to_database.py --image zhangsan.jpg --force

  # 指定底库路径（默认 face_database.npz）
  python add_to_database.py --image face.jpg --database my_db.npz
"""

import os
import sys
import argparse

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ========== 配置 ==========
DEFAULT_DB = r"C:\yolo_learning\face_database.npz"
MODEL_NAME = 'buffalo_l'
MODEL_ROOT = r'C:\Users\24522\.insightface'
DET_SIZE = (640, 640)
# ==========================


def load_database(db_path: str) -> dict:
    """加载现有底库，返回 {name: embedding}。底库不存在则返回空字典。"""
    if not os.path.exists(db_path):
        print(f"[INFO] 底库不存在，将创建新底库: {db_path}")
        return {}

    data = np.load(db_path, allow_pickle=True)
    names = data['names']
    embeddings = data['embeddings']
    database = {str(name): emb for name, emb in zip(names, embeddings)}
    print(f"[INFO] 已加载底库: {len(database)} 人")
    return database


def save_database(database: dict, db_path: str):
    """保存底库到 .npz 文件。"""
    names = np.array(list(database.keys()))
    embeddings = np.array(list(database.values()))
    np.savez(db_path, names=names, embeddings=embeddings)
    print(f"[OK] 底库已保存: {db_path}  ({len(database)} 人)")


def extract_embedding(app: FaceAnalysis, img_path: str):
    """从单张图片提取人脸特征向量。返回 (embedding, msg) — 失败时 embedding 为 None。"""
    if not os.path.exists(img_path):
        return None, f"文件不存在: {img_path}"

    img = cv2.imread(img_path)
    if img is None:
        return None, f"无法读取图片: {img_path}"

    faces = app.get(img)
    if not faces:
        return None, "未检测到人脸"

    # 取置信度最高的脸
    face = max(faces, key=lambda f: f.det_score)
    return face.embedding, f"检测到人脸 (置信度: {face.det_score:.4f})"


def add_single(app: FaceAnalysis, img_path: str, name: str,
               database: dict, force: bool) -> int:
    """添加单张图片到底库。返回实际添加数量（0 或 1）。"""
    print(f"\n{'─'*50}")
    print(f"处理: {img_path}")

    if name is None:
        name = os.path.splitext(os.path.basename(img_path))[0]

    # 重名检查
    if name in database and not force:
        print(f"[SKIP] 名字 '{name}' 已存在，跳过。使用 --force 强制覆盖")
        return 0
    elif name in database and force:
        print(f"[WARN] 覆盖已有记录: {name}")

    embedding, msg = extract_embedding(app, img_path)
    if embedding is None:
        print(f"[FAIL] {msg}")
        return 0

    database[name] = embedding
    print(f"[OK] 已添加: {name}  |  {msg}  |  特征维度: {embedding.shape}")
    return 1


def add_directory(app: FaceAnalysis, dir_path: str, database: dict,
                  force: bool) -> int:
    """批量添加目录下所有图片。返回成功添加的数量。"""
    if not os.path.isdir(dir_path):
        print(f"[ERROR] 目录不存在: {dir_path}")
        return 0

    image_files = sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print(f"[WARN] 目录下没有图片文件")
        return 0

    print(f"\n[INFO] 找到 {len(image_files)} 张图片")

    added = 0
    for filename in image_files:
        img_path = os.path.join(dir_path, filename)
        name = os.path.splitext(filename)[0]
        added += add_single(app, img_path, name, database, force)

    return added


def main():
    parser = argparse.ArgumentParser(
        description="增量添加人脸图片到底库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python add_to_database.py --image xiaoming.jpg
  python add_to_database.py --image xiaoming.jpg --name xm
  python add_to_database.py --dir ./new_faces
  python add_to_database.py --image face.jpg --force
        """,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', type=str, help="单张图片路径")
    src.add_argument('--dir', type=str, help="图片目录（批量添加）")

    parser.add_argument('--name', type=str, default=None,
                        help="人名/ID（默认使用文件名去掉扩展名）")
    parser.add_argument('--database', type=str, default=DEFAULT_DB,
                        help=f"底库路径（默认: {DEFAULT_DB}）")
    parser.add_argument('--force', action='store_true',
                        help="强制覆盖已存在的同名记录")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU ID（-1=CPU，0=第一块GPU）")

    args = parser.parse_args()

    # ---- 加载 ArcFace 模型 ----
    print("=" * 50)
    print("加载 ArcFace 模型...")
    app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT)
    app.prepare(ctx_id=args.gpu, det_size=DET_SIZE)
    print("模型加载完成")
    print("=" * 50)

    # ---- 加载现有底库 ----
    database = load_database(args.database)

    # ---- 执行添加 ----
    before_count = len(database)

    if args.image:
        added = add_single(app, args.image, args.name, database, force=args.force)
    else:
        added = add_directory(app, args.dir, database, force=args.force)

    after_count = len(database)

    # ---- 保存 ----
    if added > 0:
        save_database(database, args.database)

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print(f"增量添加完成!")
    print(f"  原有: {before_count} 人")
    print(f"  新增: {added} 人")
    print(f"  覆盖: {before_count + added - after_count} 人")
    print(f"  合计: {after_count} 人")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
