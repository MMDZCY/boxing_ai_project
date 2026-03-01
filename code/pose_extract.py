from ultralytics import YOLO
import cv2
import numpy as np
import os

# 加载预训练姿态模型（首次运行自动下载，无需手动操作）
# yolov8n-pose.pt：轻量版，CPU也能实时跑；追求精度换yolov8m-pose.pt
model = YOLO('yolov8n-pose.pt')

# 路径配置（严格对应之前的文件夹结构）
input_video_path = "../input_video/test_boxing.mp4"
output_video_path = "../output/pose_result.mp4"
output_keypoints_path = "../output/user_boxing_keypoints.npy"

# 视频读取与初始化
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

all_keypoints = []  # 存储每帧17个关键点的时序数据(x,y,置信度)

# 逐帧推理与关键点提取
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # 姿态推理，关闭冗余日志
    results = model(frame, verbose=False)
    # 绘制关键点与骨骼，生成标注视频
    annotated_frame = results[0].plot()
    
    # 提取单人关键点（确保画面只有1个测试者）
    if results[0].keypoints.data.shape[0] > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        all_keypoints.append(keypoints)
    
    # 写入输出视频，可选实时预览
    out.write(annotated_frame)
    cv2.imshow('Boxing Pose Estimation', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 资源释放
cap.release()
out.release()
cv2.destroyAllWindows()

# 保存时序关键点数据，后续所有评估都基于这个文件
np.save(output_keypoints_path, np.array(all_keypoints))
print(f"✅ 关键点提取完成，共{len(all_keypoints)}帧")
print(f"📁 标注视频保存至：{output_video_path}")
print(f"📁 关键点数据保存至：{output_keypoints_path}")