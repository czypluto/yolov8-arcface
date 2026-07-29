"""
底库人脸比对系统 — 1:N 人脸检索

支持四种输入模式:
  - image : 单张图片查询
  - batch : 批量图片查询（目录）
  - camera: 摄像头实时比对
  - video : 视频文件比对

用法:
  python face_search.py --mode image --input bus.jpg
  python face_search.py --mode batch --input C:\yolo_learning\faces\faces\images
  python face_search.py --mode camera
  python face_search.py --mode video --input video.mp4 --skip-frames 5
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

# Fix Unicode encoding on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


# ========== 工具函数 ==========

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算向量 a (D,) 与矩阵 b (N, D) 的余弦相似度。
    返回形状为 (N,) 的相似度数组，值域 [-1, 1]。
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def make_color(idx: int) -> tuple:
    """根据序号生成不同颜色 (B, G, R)，用于区分多张人脸。"""
    colors = [
        (0, 255, 0),    # 绿
        (255, 0, 0),    # 蓝
        (0, 0, 255),    # 红
        (0, 255, 255),  # 黄
        (255, 0, 255),  # 紫
        (255, 255, 0),  # 青
    ]
    return colors[idx % len(colors)]


# ========== 核心类 ==========

class FaceSearcher:
    """人脸检索器：加载底库 + 模型，提供比对功能"""

    def __init__(
        self,
        database_path: str = r"C:\yolo_learning\face_database.npz",
        det_size: tuple = (640, 640),
        threshold: float = 0.7,
        ctx_id: int = -1,
    ):
        """
        Args:
            database_path: 底库 .npz 文件路径
            det_size: 人脸检测分辨率
            threshold: 相似度阈值，低于此值视为 Unknown
            ctx_id: GPU ID（-1 为 CPU，0+ 为 GPU）
        """
        self.threshold = threshold
        self.det_size = det_size

        # ---- 1. 加载 InsightFace 模型 ----
        print("=" * 60)
        print("加载 InsightFace 模型...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=r'C:\Users\24522\.insightface',
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("模型加载完成")

        # ---- 2. 加载底库 ----
        print(f"\n加载底库: {database_path}")
        data = np.load(database_path, allow_pickle=True)
        self.db_names: np.ndarray = data['names']          # (N,)
        self.db_embeddings: np.ndarray = data['embeddings']  # (N, 512)
        self.db_size = len(self.db_names)
        print(f"底库人数: {self.db_size}")
        print(f"特征维度: {self.db_embeddings.shape[1]}")
        print(f"相似度阈值: {self.threshold}")
        print("=" * 60 + "\n")

        # 收集所有结果用于最终导出报告
        self.all_results = []

    # ---- 核心匹配 ----

    def search(self, embedding: np.ndarray) -> dict:
        """
        对单个 embedding 进行 1:N 检索，返回最佳匹配。

        Returns:
            dict: {
                'name': 匹配人名或 'Unknown',
                'similarity': 余弦相似度,
                'matched': 是否高于阈值,
            }
        """
        sims = cosine_similarity(embedding, self.db_embeddings)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            return {
                'name': str(self.db_names[best_idx]),
                'similarity': best_sim,
                'matched': True,
                'db_index': best_idx,
            }
        else:
            return {
                'name': 'Unknown',
                'similarity': best_sim,
                'matched': False,
                'db_index': best_idx,
            }

    # ---- 绘制结果 ----

    def draw_results(self, img: np.ndarray, faces: list, matches: list[dict]) -> np.ndarray:
        """在图片上画人脸框 + 匹配结果。"""
        img_out = img.copy()
        h, w = img_out.shape[:2]

        for i, (face, match) in enumerate(zip(faces, matches)):
            color = make_color(i)
            x1, y1, x2, y2 = map(int, face.bbox)

            # 画框
            thickness = max(2, int(min(w, h) / 300))
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)

            # 标签文字
            label = f"{match['name']}  {match['similarity']:.3f}"
            font_scale = max(0.5, min(w, h) / 800)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 标签背景
            label_y = max(y1 - th - 10, 0)
            cv2.rectangle(img_out,
                          (x1, label_y),
                          (x1 + tw + 10, y1),
                          color, -1)

            # 标签文字
            cv2.putText(img_out, label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness)

            # 如果未匹配，额外标注
            if not match['matched']:
                cv2.putText(img_out, "NO MATCH",
                            (x1 + 5, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale * 1.2, (0, 0, 255), thickness * 2)

        return img_out

    # ---- 单张图片 ----

    def process_image(self, img_path: str, output_dir: str = None) -> list[dict]:
        """
        处理单张图片。保存标注后的结果图。

        Returns:
            每张人脸一条结果记录
        """
        print(f"\n{'='*60}")
        print(f"查询图片: {img_path}")
        print(f"{'='*60}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取 {img_path}")
            return []

        faces = self.app.get(img)
        print(f"检测到 {len(faces)} 张人脸")

        if not faces:
            return []

        results = []
        matches = []
        for i, face in enumerate(faces):
            match = self.search(face.embedding)
            matches.append(match)

            rec = {
                'image': os.path.basename(img_path),
                'face_index': i,
                'bbox': [round(float(v), 1) for v in face.bbox],
                'det_score': round(float(face.det_score), 4),
                'matched_name': match['name'],
                'similarity': round(match['similarity'], 4),
                'matched': match['matched'],
            }
            results.append(rec)
            self.all_results.append(rec)

            # 控制台输出
            status = "[MATCH]" if match['matched'] else "[NO MATCH]"
            print(f"  Face {i+1}: {status} {match['name']}  "
                  f"(similarity: {match['similarity']:.4f}, det_score: {face.det_score:.4f})")

        # 保存标注图片
        img_out = self.draw_results(img, faces, matches)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_name = os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
            out_path = os.path.join(output_dir, out_name)
        else:
            out_path = os.path.splitext(img_path)[0] + "_result.jpg"
        cv2.imwrite(out_path, img_out)
        print(f"\n结果图已保存: {out_path}")

        return results

    # ---- 批量图片 ----

    def process_batch(self, dir_path: str, output_dir: str = None) -> list[dict]:
        """
        批量处理一个目录下的所有图片。

        Returns:
            所有图片的所有人脸结果
        """
        print(f"\n{'='*60}")
        print(f"批量查询目录: {dir_path}")
        print(f"{'='*60}")

        image_files = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"找到 {len(image_files)} 张图片\n")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_results = []
        for img_path in tqdm(image_files, desc="批量比对"):
            results = self.process_image(img_path, output_dir=output_dir)
            all_results.extend(results)

        # 汇总统计
        matched = [r for r in all_results if r['matched']]
        print(f"\n{'='*60}")
        print(f"批量比对完成!")
        print(f"  总图片数:   {len(image_files)}")
        print(f"  总人脸数:   {len(all_results)}")
        print(f"  匹配成功:   {len(matched)}")
        print(f"  匹配失败:   {len(all_results) - len(matched)}")
        if matched:
            avg_sim = np.mean([r['similarity'] for r in matched])
            print(f"  平均相似度: {avg_sim:.4f}")
        print(f"{'='*60}")

        return all_results

    # ---- 摄像头实时比对 ----

    def process_camera(self, camera_id: int = 0, output_dir: str = None,
                        skip_frames: int = 2, cam_det_size: int = 320):
        """
        打开摄像头实时人脸比对。按 'q' 退出，按 's' 截图保存。

        Args:
            camera_id: 摄像头设备 ID
            output_dir: 截图保存目录
            skip_frames: 每隔多少帧做一次检测（默认 2，即每 3 帧检测 1 帧）
            cam_det_size: 摄像头检测分辨率（默认 320，比默认 640 快约 4 倍）
        """
        print(f"\n{'='*60}")
        print(f"摄像头实时比对 (ID: {camera_id})")
        print(f"检测分辨率: {cam_det_size}x{cam_det_size}")
        print(f"跳帧: 每 {skip_frames + 1} 帧检测 1 帧")
        print(f"按 'q' 退出  按 's' 截图保存")
        print(f"{'='*60}")

        # ---- 为摄像头创建轻量级模型（只加载检测+识别，跳过性别/年龄/关键点）----
        print("加载摄像头专用轻量模型...")
        cam_app = FaceAnalysis(
            name='buffalo_l',
            root=r'C:\Users\24522\.insightface',
            models=['detection', 'recognition'],  # 只保留必需的 2 个模型
        )
        cam_app.prepare(ctx_id=-1, det_size=(cam_det_size, cam_det_size))
        print(f"轻量模型加载完成\n")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_id}")
            return

        # 降低摄像头采集分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        saved_count = 0

        # 缓存上次检测结果
        last_faces = []
        last_matches = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                frame_count += 1

                # 跳帧
                do_detect = (skip_frames == 0 or frame_count % (skip_frames + 1) == 1)

                if do_detect:
                    last_faces = cam_app.get(frame)
                    last_matches = []
                    for i, face in enumerate(last_faces):
                        match = self.search(face.embedding)
                        last_matches.append(match)

                        rec = {
                            'image': f'camera_frame_{frame_count}',
                            'face_index': i,
                            'bbox': [round(float(v), 1) for v in face.bbox],
                            'det_score': round(float(face.det_score), 4),
                            'matched_name': match['name'],
                            'similarity': round(match['similarity'], 4),
                            'matched': match['matched'],
                        }
                        self.all_results.append(rec)

                # 画结果（用缓存的检测结果）
                frame_out = self.draw_results(frame, last_faces, last_matches)

                status = "DETECT" if do_detect else "CACHED"
                cv2.putText(frame_out,
                            f"Faces: {len(last_faces)} | {status} | Frame: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                cv2.imshow("Face Search - Camera", frame_out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s'):
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        save_path = os.path.join(output_dir, f"capture_{saved_count:04d}.jpg")
                    else:
                        save_path = f"capture_{saved_count:04d}.jpg"
                    cv2.imwrite(save_path, frame_out)
                    saved_count += 1
                    print(f"  截图已保存: {save_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"摄像头已关闭。共处理 {frame_count} 帧，保存 {saved_count} 张截图")

    # ---- 视频文件比对 ----

    def process_video(self, video_path: str, output_dir: str = None,
                      skip_frames: int = 0, save_video: bool = True):
        """
        处理视频文件，逐帧（或跳帧）比对。

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            skip_frames: 每隔多少帧检测一次（0 = 每帧检测）
            save_video: 是否输出带标注的视频
        """
        print(f"\n{'='*60}")
        print(f"视频文件比对: {video_path}")
        print(f"跳帧设置: 每 {skip_frames + 1} 帧检测 1 帧")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"视频信息: {width}x{height}, {fps:.1f} fps, {total_frames} 帧")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 输出视频
        video_writer = None
        if save_video and output_dir:
            out_video_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(video_path))[0] + "_result.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
            print(f"输出视频: {out_video_path}")

        frame_count = 0
        processed_count = 0
        pbar = tqdm(total=total_frames, desc="视频处理")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                # 跳帧
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    if video_writer:
                        video_writer.write(frame)
                    continue

                processed_count += 1

                # 检测 + 比对
                faces = self.app.get(frame)
                matches = []
                for i, face in enumerate(faces):
                    match = self.search(face.embedding)
                    matches.append(match)

                    rec = {
                        'image': os.path.basename(video_path),
                        'face_index': i,
                        'frame': frame_count,
                        'bbox': [round(float(v), 1) for v in face.bbox],
                        'det_score': round(float(face.det_score), 4),
                        'matched_name': match['name'],
                        'similarity': round(match['similarity'], 4),
                        'matched': match['matched'],
                    }
                    self.all_results.append(rec)

                frame_out = self.draw_results(frame, faces, matches)

                if video_writer:
                    video_writer.write(frame_out)

        finally:
            pbar.close()
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        print(f"\n视频处理完成! 共 {frame_count} 帧, 处理 {processed_count} 帧")
        if video_writer:
            print(f"输出视频已保存: {out_video_path}")

    # ---- 报告导出 ----

    def export_report(self, output_dir: str = "."):
        """将所有结果导出为 CSV 和 JSON 报告。"""
        if not self.all_results:
            print("没有结果可导出")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = os.path.join(output_dir, f"face_search_{timestamp}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_results[0].keys())
            writer.writeheader()
            writer.writerows(self.all_results)
        print(f"CSV 报告已保存: {csv_path}")

        # JSON
        json_path = os.path.join(output_dir, f"face_search_{timestamp}.json")
        summary = {
            'timestamp': timestamp,
            'total_faces': len(self.all_results),
            'matched': sum(1 for r in self.all_results if r['matched']),
            'unmatched': sum(1 for r in self.all_results if not r['matched']),
            'results': self.all_results,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"JSON 报告已保存: {json_path}")

        # 简短汇总
        print(f"\n{'='*60}")
        print(f"比对报告汇总:")
        print(f"  总检测人脸: {summary['total_faces']}")
        print(f"  匹配成功:   {summary['matched']}")
        print(f"  匹配失败:   {summary['unmatched']}")
        if summary['total_faces'] > 0:
            rate = summary['matched'] / summary['total_faces'] * 100
            print(f"  匹配率:     {rate:.1f}%")
        print(f"{'='*60}")


# ========== 命令行入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="底库人脸比对系统 — 支持单图/批量/摄像头/视频四种模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python face_search.py --mode image --input bus.jpg
  python face_search.py --mode batch --input C:\\yolo_learning\\faces\\faces\\images
  python face_search.py --mode camera
  python face_search.py --mode video --input video.mp4 --skip-frames 5
        """,
    )
    parser.add_argument('--mode', required=True,
                        choices=['image', 'batch', 'camera', 'video'],
                        help="输入模式")
    parser.add_argument('--input', type=str, default=None,
                        help="输入路径：图片/目录/视频文件路径（camera 模式不需要）")
    parser.add_argument('--threshold', type=float, default=0.7,
                        help="相似度阈值，低于此值标记为 Unknown（默认 0.7）")
    parser.add_argument('--output-dir', type=str, default=r"C:\yolo_learning\search_results",
                        help="结果输出目录（默认: C:\\yolo_learning\\search_results）")
    parser.add_argument('--database', type=str,
                        default=r"C:\yolo_learning\face_database.npz",
                        help="底库路径")
    parser.add_argument('--det-size', type=int, default=640,
                        help="检测分辨率（默认 640）")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU ID（-1=CPU，0=GPU0，默认 -1）")
    parser.add_argument('--skip-frames', type=int, default=0,
                        help="视频模式跳帧数（0=每帧检测，越大越快但可能漏检）")
    parser.add_argument('--no-export', action='store_true',
                        help="不导出 CSV/JSON 报告")
    parser.add_argument('--no-save-video', action='store_true',
                        help="视频模式不输出带标注的视频文件")
    parser.add_argument('--cam-det-size', type=int, default=320,
                        help="摄像头检测分辨率（默认 320，越小越快但检出率越低）")

    args = parser.parse_args()

    # 验证参数
    if args.mode in ('image', 'batch', 'video') and not args.input:
        parser.error(f"--mode {args.mode} 需要指定 --input")

    # 初始化
    searcher = FaceSearcher(
        database_path=args.database,
        det_size=(args.det_size, args.det_size),
        threshold=args.threshold,
        ctx_id=args.gpu,
    )

    start_time = time.time()

    # 分发模式
    if args.mode == 'image':
        searcher.process_image(args.input, output_dir=args.output_dir)

    elif args.mode == 'batch':
        searcher.process_batch(args.input, output_dir=args.output_dir)

    elif args.mode == 'camera':
        searcher.process_camera(
            output_dir=args.output_dir,
            skip_frames=args.skip_frames,
            cam_det_size=args.cam_det_size,
        )

    elif args.mode == 'video':
        searcher.process_video(
            args.input,
            output_dir=args.output_dir,
            skip_frames=args.skip_frames,
            save_video=not args.no_save_video,
        )

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")

    # 导出报告
    if not args.no_export and searcher.all_results:
        searcher.export_report(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
