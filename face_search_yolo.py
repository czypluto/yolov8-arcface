"""
YOLO + ArcFace 联合人脸比对系统

管线:
  YOLO 检测人脸 bbox → SCRFD 在裁剪区域提取 5 点关键点
  → norm_crop 对齐 → ArcFaceONNX 提取 512 维 embedding
  → 与底库余弦相似度比对 → Top-1 匹配

用法:
  python face_search_yolo.py --mode image --input bus.jpg
  python face_search_yolo.py --mode batch --input C:\yolo_learning\faces\faces\images
  python face_search_yolo.py --mode camera
  python face_search_yolo.py --mode video --input video.mp4 --skip-frames 5
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from insightface.model_zoo import SCRFD, ArcFaceONNX
from insightface.utils.face_align import norm_crop


# ========== 常量 ==========
MODEL_ROOT = r'C:\Users\24522\.insightface\models\buffalo_l'
DET_MODEL  = os.path.join(MODEL_ROOT, 'det_10g.onnx')
REC_MODEL  = os.path.join(MODEL_ROOT, 'w600k_r50.onnx')
YOLO_MODEL = r'C:\yolo_learning\face_dataset\train\weights\best.pt'
DB_PATH    = r'C:\yolo_learning\face_database.npz'
ARC_INPUT_SIZE = (112, 112)  # ArcFace 识别模型输入尺寸


# ========== 工具函数 ==========

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def make_color(idx: int) -> tuple:
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
    ]
    return colors[idx % len(colors)]


def expand_bbox(bbox, margin, img_w, img_h):
    """扩展 bbox 边界，确保人脸完整保留用于关键点检测。"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(img_w, int(x2 + bw * margin))
    y2 = min(img_h, int(y2 + bh * margin))
    return x1, y1, x2, y2


# ========== 核心类 ==========

class FaceSearcherYolo:
    """YOLO 检测 + ArcFace 识别 + 底库比对"""

    def __init__(
        self,
        yolo_path: str = YOLO_MODEL,
        det_model: str = DET_MODEL,
        rec_model: str = REC_MODEL,
        database_path: str = DB_PATH,
        threshold: float = 0.7,
        yolo_conf: float = 0.5,
        scrfd_det_size: int = 320,
        ctx_id: int = -1,
    ):
        self.threshold = threshold
        self.yolo_conf = yolo_conf

        # ---- 1. 加载 YOLO ----
        print("=" * 60)
        print(f"加载 YOLO 模型: {yolo_path}")
        self.yolo = YOLO(yolo_path)
        print("YOLO 加载完成")

        # ---- 2. 加载 SCRFD（仅用于关键点提取）----
        print(f"加载 SCRFD 关键点模型: {det_model}")
        self.scrfd = SCRFD(det_model)
        self.scrfd.prepare(ctx_id=ctx_id)
        self.scrfd_det_size = scrfd_det_size
        print(f"SCRFD 加载完成 (det_size={scrfd_det_size})")

        # ---- 3. 加载 ArcFace 识别模型 ----
        print(f"加载 ArcFace 识别模型: {rec_model}")
        self.rec = ArcFaceONNX(rec_model)
        self.rec.prepare(ctx_id=ctx_id)
        print(f"ArcFace 加载完成 (input_size={self.rec.input_size})")

        # ---- 4. 加载底库 ----
        print(f"\n加载底库: {database_path}")
        data = np.load(database_path, allow_pickle=True)
        self.db_names: np.ndarray = data['names']
        self.db_embeddings: np.ndarray = data['embeddings']
        self.db_size = len(self.db_names)
        print(f"底库人数: {self.db_size}")
        print(f"特征维度: {self.db_embeddings.shape[1]}")
        print(f"相似度阈值: {self.threshold}")
        print(f"YOLO 置信度: {self.yolo_conf}")
        print("=" * 60 + "\n")

        self.all_results = []

    # ---- YOLO 检测 ----

    def detect_faces(self, img: np.ndarray):
        """YOLO 检测人脸，返回 bbox 列表 + 置信度列表。"""
        results = self.yolo(img, verbose=False, conf=self.yolo_conf)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []

        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(bboxes))
        return bboxes, confs

    # ---- 单张人脸处理 ----

    def process_face(self, img: np.ndarray, bbox: np.ndarray) -> dict:
        """
        对单个 YOLO bbox 区域做人脸关键点 + 对齐 + embedding 提取。

        Args:
            img: 全图 (H, W, 3)
            bbox: [x1, y1, x2, y2]（YOLO 输出）

        Returns:
            dict: {'embedding': np.ndarray, 'kps': np.ndarray, 'scrfd_score': float}
                  如果提取失败返回 None
        """
        h, w = img.shape[:2]

        # 扩展 bbox 边界，确保包含完整人脸
        x1, y1, x2, y2 = expand_bbox(bbox, margin=0.2, img_w=w, img_h=h)
        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = img[y1:y2, x1:x2]

        # 在裁剪区域运行 SCRFD 获取关键点
        try:
            bboxes_kp, kpss = self.scrfd.detect(
                face_crop, input_size=(self.scrfd_det_size, self.scrfd_det_size)
            )
        except Exception:
            return None

        if len(bboxes_kp) == 0:
            return None

        # 取置信度最高的检测结果
        best_idx = np.argmax(bboxes_kp[:, 4])
        kps_crop = kpss[best_idx]        # (5, 2)，相对于 crop 的坐标
        scrfd_score = float(bboxes_kp[best_idx, 4])

        # 关键点坐标映射回原图
        kps_global = kps_crop.copy()
        kps_global[:, 0] += x1
        kps_global[:, 1] += y1

        # norm_crop 对齐
        aligned = norm_crop(img, landmark=kps_global, image_size=ARC_INPUT_SIZE[0])

        # 提取 embedding
        embedding = self.rec.get_feat(aligned).flatten()

        return {
            'embedding': embedding,
            'kps': kps_global,
            'scrfd_score': scrfd_score,
        }

    # ---- 底库比对 ----

    def search(self, embedding: np.ndarray) -> dict:
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

    # ---- 绘制 ----

    def draw_results(self, img, faces_info) -> np.ndarray:
        """faces_info: list of dict with bbox, match, yolo_conf"""
        img_out = img.copy()
        h, w = img_out.shape[:2]

        for i, fi in enumerate(faces_info):
            color = make_color(i)
            x1, y1, x2, y2 = map(int, fi['bbox'])
            match = fi['match']

            thickness = max(2, int(min(w, h) / 300))
            # 绿框=匹配成功，红框=Unknown
            box_color = (0, 255, 0) if match['matched'] else (0, 0, 255)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), box_color, thickness)

            label = f"{match['name']}  {match['similarity']:.3f}"
            font_scale = max(0.5, min(w, h) / 800)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            label_y = max(y1 - th - 10, 0)
            cv2.rectangle(img_out, (x1, label_y), (x1 + tw + 10, y1), box_color, -1)
            cv2.putText(img_out, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # 画关键点（小绿点）
            if fi.get('kps') is not None:
                for kp in fi['kps']:
                    cv2.circle(img_out, (int(kp[0]), int(kp[1])), 2, color, -1)

        return img_out

    # ---- 单张图片 ----

    def process_image(self, img_path: str, output_dir: str = None) -> list[dict]:
        print(f"\n{'='*60}")
        print(f"查询图片: {img_path}")
        print(f"{'='*60}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取 {img_path}")
            return []

        bboxes, confs = self.detect_faces(img)
        print(f"YOLO 检测到 {len(bboxes)} 张人脸")

        if len(bboxes) == 0:
            return []

        results = []
        faces_info = []
        for i, (bbox, conf) in enumerate(zip(bboxes, confs)):
            face_data = self.process_face(img, bbox)
            if face_data is None:
                print(f"  Face {i+1}: SCRFD 关键点提取失败，跳过")
                continue

            match = self.search(face_data['embedding'])

            rec = {
                'image': os.path.basename(img_path),
                'face_index': i,
                'bbox': [round(float(v), 1) for v in bbox],
                'yolo_conf': round(float(conf), 4),
                'scrfd_score': round(face_data['scrfd_score'], 4),
                'matched_name': match['name'],
                'similarity': round(match['similarity'], 4),
                'matched': match['matched'],
            }
            results.append(rec)
            self.all_results.append(rec)

            faces_info.append({
                'bbox': bbox,
                'match': match,
                'yolo_conf': conf,
                'kps': face_data['kps'],
            })

            status = "[MATCH]" if match['matched'] else "[NO MATCH]"
            print(f"  Face {i+1}: {status} {match['name']}  "
                  f"(similarity: {match['similarity']:.4f}, yolo_conf: {conf:.4f})")

        # 保存标注图片
        if faces_info:
            img_out = self.draw_results(img, faces_info)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_name = os.path.splitext(os.path.basename(img_path))[0] + "_yolo_result.jpg"
                out_path = os.path.join(output_dir, out_name)
            else:
                out_path = os.path.splitext(img_path)[0] + "_yolo_result.jpg"
            cv2.imwrite(out_path, img_out)
            print(f"\n结果图已保存: {out_path}")

        return results

    # ---- 批量图片 ----

    def process_batch(self, dir_path: str, output_dir: str = None) -> list[dict]:
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

        matched = [r for r in all_results if r['matched']]
        print(f"\n{'='*60}")
        print(f"批量比对完成!")
        print(f"  总图片数:   {len(image_files)}")
        print(f"  总人脸数:   {len(all_results)}")
        print(f"  匹配成功:   {len(matched)}")
        print(f"  匹配失败:   {len(all_results) - len(matched)}")
        if matched:
            print(f"  平均相似度: {np.mean([r['similarity'] for r in matched]):.4f}")
        print(f"{'='*60}")

        return all_results

    # ---- 摄像头 ----

    def process_camera(self, camera_id: int = 0, output_dir: str = None,
                       skip_frames: int = 2):
        print(f"\n{'='*60}")
        print(f"摄像头实时比对 (ID: {camera_id})")
        print(f"检测: YOLOv8 + 关键点: SCRFD({self.scrfd_det_size}) + 识别: ArcFace")
        print(f"跳帧: 每 {skip_frames + 1} 帧检测 1 帧")
        print(f"按 'q' 退出  按 's' 截图保存")
        print(f"{'='*60}\n")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        saved_count = 0
        last_faces_info = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                frame_count += 1
                do_detect = (skip_frames == 0 or frame_count % (skip_frames + 1) == 1)

                if do_detect:
                    bboxes, confs = self.detect_faces(frame)
                    last_faces_info = []
                    for bbox, conf in zip(bboxes, confs):
                        face_data = self.process_face(frame, bbox)
                        if face_data is None:
                            # 即使关键点提取失败也画出 YOLO bbox
                            last_faces_info.append({
                                'bbox': bbox,
                                'match': {'name': '?', 'similarity': 0.0, 'matched': False},
                                'yolo_conf': conf,
                                'kps': None,
                            })
                            continue

                        match = self.search(face_data['embedding'])
                        last_faces_info.append({
                            'bbox': bbox,
                            'match': match,
                            'yolo_conf': conf,
                            'kps': face_data['kps'],
                        })

                        rec = {
                            'image': f'camera_frame_{frame_count}',
                            'face_index': len(last_faces_info) - 1,
                            'bbox': [round(float(v), 1) for v in bbox],
                            'yolo_conf': round(float(conf), 4),
                            'matched_name': match['name'],
                            'similarity': round(match['similarity'], 4),
                            'matched': match['matched'],
                        }
                        self.all_results.append(rec)

                frame_out = self.draw_results(frame, last_faces_info)

                status = "DETECT" if do_detect else "CACHED"
                cv2.putText(frame_out,
                            f"Faces: {len(last_faces_info)} | {status} | Frame: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                cv2.imshow("Face Search - YOLO+ArcFace", frame_out)

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
            print(f"摄像头已关闭。共 {frame_count} 帧，保存 {saved_count} 张截图")

    # ---- 视频文件 ----

    def process_video(self, video_path: str, output_dir: str = None,
                      skip_frames: int = 0, save_video: bool = True):
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

        video_writer = None
        if save_video and output_dir:
            out_video_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(video_path))[0] + "_yolo_result.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

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

                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    if video_writer:
                        video_writer.write(frame)
                    continue

                processed_count += 1
                bboxes, confs = self.detect_faces(frame)
                faces_info = []
                for bbox, conf in zip(bboxes, confs):
                    face_data = self.process_face(frame, bbox)
                    if face_data is None:
                        faces_info.append({
                            'bbox': bbox,
                            'match': {'name': '?', 'similarity': 0.0, 'matched': False},
                            'yolo_conf': conf,
                            'kps': None,
                        })
                        continue

                    match = self.search(face_data['embedding'])
                    faces_info.append({
                        'bbox': bbox,
                        'match': match,
                        'yolo_conf': conf,
                        'kps': face_data['kps'],
                    })

                    rec = {
                        'image': os.path.basename(video_path),
                        'face_index': len(faces_info) - 1,
                        'frame': frame_count,
                        'bbox': [round(float(v), 1) for v in bbox],
                        'yolo_conf': round(float(conf), 4),
                        'matched_name': match['name'],
                        'similarity': round(match['similarity'], 4),
                        'matched': match['matched'],
                    }
                    self.all_results.append(rec)

                frame_out = self.draw_results(frame, faces_info)
                if video_writer:
                    video_writer.write(frame_out)

        finally:
            pbar.close()
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        print(f"\n视频处理完成! 共 {frame_count} 帧, 处理 {processed_count} 帧")

    # ---- 报告导出 ----

    def export_report(self, output_dir: str = "."):
        if not self.all_results:
            print("没有结果可导出")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(output_dir, f"face_search_yolo_{timestamp}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_results[0].keys())
            writer.writeheader()
            writer.writerows(self.all_results)
        print(f"CSV 报告已保存: {csv_path}")

        json_path = os.path.join(output_dir, f"face_search_yolo_{timestamp}.json")
        summary = {
            'timestamp': timestamp,
            'detector': 'YOLOv8',
            'recognizer': 'ArcFace (w600k_r50)',
            'total_faces': len(self.all_results),
            'matched': sum(1 for r in self.all_results if r['matched']),
            'unmatched': sum(1 for r in self.all_results if not r['matched']),
            'results': self.all_results,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"JSON 报告已保存: {json_path}")

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
        description="YOLO + ArcFace 联合人脸比对系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python face_search_yolo.py --mode image --input bus.jpg
  python face_search_yolo.py --mode batch --input C:\\yolo_learning\\faces\\faces\\images
  python face_search_yolo.py --mode camera
  python face_search_yolo.py --mode video --input video.mp4 --skip-frames 5
        """,
    )
    parser.add_argument('--mode', required=True,
                        choices=['image', 'batch', 'camera', 'video'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.7,
                        help="相似度阈值（默认 0.7）")
    parser.add_argument('--yolo-conf', type=float, default=0.5,
                        help="YOLO 检测置信度阈值（默认 0.5）")
    parser.add_argument('--output-dir', type=str, default=r"C:\yolo_learning\search_results")
    parser.add_argument('--database', type=str, default=DB_PATH)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--skip-frames', type=int, default=2,
                        help="跳帧数（0=每帧检测，默认 2）")
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--no-save-video', action='store_true')

    args = parser.parse_args()

    if args.mode in ('image', 'batch', 'video') and not args.input:
        parser.error(f"--mode {args.mode} 需要指定 --input")

    searcher = FaceSearcherYolo(
        database_path=args.database,
        threshold=args.threshold,
        yolo_conf=args.yolo_conf,
        ctx_id=args.gpu,
    )

    start_time = time.time()

    if args.mode == 'image':
        searcher.process_image(args.input, output_dir=args.output_dir)
    elif args.mode == 'batch':
        searcher.process_batch(args.input, output_dir=args.output_dir)
    elif args.mode == 'camera':
        searcher.process_camera(output_dir=args.output_dir, skip_frames=args.skip_frames)
    elif args.mode == 'video':
        searcher.process_video(
            args.input, output_dir=args.output_dir,
            skip_frames=args.skip_frames, save_video=not args.no_save_video,
        )

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")

    if not args.no_export and searcher.all_results:
        searcher.export_report(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
