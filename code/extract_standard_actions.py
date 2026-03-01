from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO('yolov8n-pose.pt')
standard_folder = "../standard_action"

# 遍历文件夹里的所有mp4视频
for filename in os.listdir(standard_folder):
    if filename.endswith(".mp4") and filename.startswith("standard_"):
        video_path = os.path.join(standard_folder, filename)
        output_npy_path = os.path.join(standard_folder, filename.replace(".mp4", ".npy"))
        
        print(f"正在处理标准动作：{filename} ...")
        
        cap = cv2.VideoCapture(video_path)
        all_keypoints = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model(frame, verbose=False)
            if results[0].keypoints.data.shape[0] > 0:
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                all_keypoints.append(keypoints)
        
        cap.release()
        np.save(output_npy_path, np.array(all_keypoints))
        print(f"✅ 完成！标准动作模板已保存至：{output_npy_path}")

print("\n🎉 所有标准动作模板提取完成！")