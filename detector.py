# detector.py - YOLO 自动捕捉视频中的人
from ultralytics import YOLO
import cv2


class PersonDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect_persons(self, frame, conf=0.35):
        """检测视频帧中的所有人物"""
        results = self.model.predict(frame, conf=conf, verbose=False)
        persons = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            if cls_id == 0:  # person
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                persons.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(box.conf.item())
                })
        return persons


# 视频处理示例
if __name__ == "__main__":
    detector = PersonDetector()
    cap = cv2.VideoCapture("che.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        persons = detector.detect_persons(frame)
        for p in persons:
            x1, y1, x2, y2 = map(int, p['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {p['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()