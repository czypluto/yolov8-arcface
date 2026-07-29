import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))

img = cv2.imread('bus.jpg')
faces = app.get(img)
print(f'检测到 {len(faces)} 张人脸')
print()

for i, face in enumerate(faces):
    sex_str = '男' if face.sex == 'M' else '女'
    print(f'--- 人脸 {i+1} ---')
    print(f'  边界框: {face.bbox.astype(int)}')
    print(f'  置信度: {face.det_score:.4f}')
    print(f'  特征维度: {face.embedding.shape}')
    print(f'  性别: {sex_str}  年龄: {face.age}')
    print()

# 保存标注后的图片
for face in faces:
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{face.sex} {face.age}'
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite('arcface_bus_result.jpg', img)
print('标注结果已保存到 arcface_bus_result.jpg')
