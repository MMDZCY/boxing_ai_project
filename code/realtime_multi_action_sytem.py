import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# -------------------------- 配置项（不用改，适配你的路径） --------------------------
# YOLOv8-pose模型（自动下载，第一次运行会慢一点）
POSE_MODEL_PATH = "yolov8n-pose.pt"
# 标准动作模板路径
STANDARD_ACTION_PATH = "../standard_action/"
# 优化后的分类模型路径
CLASSIFIER_PATH = "../model/action_classifier_optimized.pkl"
# 动作名称映射
ACTION_MAP = {
    0: "jab(刺拳)",
    1: "straight(直拳)",
    2: "hook(勾拳)",
    3: "swing(摆拳)"
}
# 骨骼连接点（COCO 17关键点）
SKELETON_CONNECTIONS = [
    (0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]
# 窗口大小
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# -------------------------- 初始化 --------------------------
# 1. 加载YOLOv8-pose模型
print("🔄 加载YOLOv8-pose模型...")
pose_model = YOLO(POSE_MODEL_PATH)

# 2. 加载标准动作模板
print("🔄 加载标准动作模板...")
standard_actions = {}
try:
    standard_actions[0] = np.load(f"{STANDARD_ACTION_PATH}standard_jab.npy")
    standard_actions[1] = np.load(f"{STANDARD_ACTION_PATH}standard_straight.npy")
    standard_actions[2] = np.load(f"{STANDARD_ACTION_PATH}standard_hook.npy")
    standard_actions[3] = np.load(f"{STANDARD_ACTION_PATH}standard_swing.npy")
    print("✅ 标准模板加载成功！")
except Exception as e:
    print(f"⚠️  标准模板加载失败：{e}")
    print("❗ 请确保standard_action文件夹里有4个.npy文件")

# 3. 加载分类模型和标准化器
print("🔄 加载动作分类模型...")
model, scaler = None, None
try:
    model, scaler = joblib.load(CLASSIFIER_PATH)
    print("✅ 分类模型加载成功！")
except Exception as e:
    print(f"⚠️  分类模型加载失败：{e}")
    print("❗ 请先运行train_action_classifier_optimized.py训练模型")

# 4. 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

# 5. 初始化变量
current_action_id = 0  # 默认显示刺拳模板
frame_buffer = []      # 帧缓存，用于特征提取
buffer_size = 10       # 和训练时的窗口大小一致
similarity_score = 0   # 相似度打分
pred_action = "未识别" # 预测的动作名称

# -------------------------- 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    """计算三个点的夹角（度数）"""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_classify_feature(kp_seq):
    """提取8维特征（和训练时一致）"""
    window_features = []
    for kp in kp_seq:
        left_elbow = calculate_angle(kp[5], kp[7], kp[9])
        right_elbow = calculate_angle(kp[6], kp[8], kp[10])
        left_shoulder = calculate_angle(kp[11], kp[5], kp[7])
        right_shoulder = calculate_angle(kp[12], kp[6], kp[8])
        shoulder_twist = calculate_angle(np.array([0,0]), kp[6][:2]-kp[5][:2], kp[12][:2]-kp[11][:2])
        left_arm_vertical = calculate_angle(kp[7], kp[9], np.array([kp[9][0], 0]))
        right_arm_vertical = calculate_angle(kp[8], kp[10], np.array([kp[10][0], 0]))
        wrist_diff = (kp[9][1] - kp[10][1]) / (np.linalg.norm(kp[5]-kp[6]) + 1e-6)
        
        window_features.append([
            left_elbow, right_elbow, left_shoulder, right_shoulder,
            shoulder_twist, left_arm_vertical, right_arm_vertical, wrist_diff
        ])
    return np.mean(window_features, axis=0)

def predict_action(kp_seq):
    """动作分类预测（适配优化模型）"""
    if model is None or scaler is None:
        return "模型未加载"
    
    try:
        # 提取8维特征
        feature = extract_classify_feature(kp_seq)
        # 标准化
        feature_scaled = scaler.transform(feature.reshape(1, -1))
        # 特征增强（和训练时一致）
        arm_angle_diff = feature_scaled[0,0] - feature_scaled[0,1]
        shoulder_elbow_ratio = (feature_scaled[0,2]+feature_scaled[0,3])/(feature_scaled[0,0]+feature_scaled[0,1]+1e-6)
        vertical_sum = feature_scaled[0,5] + feature_scaled[0,6]
        wrist_abs = np.abs(feature_scaled[0,7])
        # 拼接增强特征
        feature_enhanced = np.hstack([
            feature_scaled,
            np.array([arm_angle_diff, shoulder_elbow_ratio, vertical_sum, wrist_abs]).reshape(1, -1)
        ])
        # 预测
        pred_id = model.predict(feature_enhanced)[0]
        return ACTION_MAP.get(pred_id, "未知动作")
    except Exception as e:
        return f"识别失败：{str(e)[:10]}"

def calculate_similarity(user_kp_seq, standard_kp_seq):
    """DTW计算动作相似度（0-100分）"""
    if len(user_kp_seq) < 5 or len(standard_kp_seq) < 5:
        return 0
    
    try:
        # 只取关键点的坐标部分，展平
        user_flat = [kp.flatten() for kp in user_kp_seq]
        standard_flat = [kp.flatten() for kp in standard_kp_seq[:len(user_kp_seq)]]
        # DTW计算距离
        distance, _ = fastdtw(user_flat, standard_flat, dist=euclidean)
        # 归一化到0-100分
        max_distance = np.sqrt(len(user_flat[0])) * len(user_flat)
        score = max(0, 100 - (distance / max_distance) * 100)
        return round(score, 1)
    except Exception as e:
        return 0

def draw_skeleton(frame, kp, color, scale=1.0, offset=(0,0)):
    """绘制骨骼（用户红/标准绿）"""
    h, w = frame.shape[:2]
    # 遍历所有连接点
    for (p1, p2) in SKELETON_CONNECTIONS:
        # 关键点坐标（归一化→像素）
        x1 = int(kp[p1][0] * w * scale + offset[0])
        y1 = int(kp[p1][1] * h * scale + offset[1])
        x2 = int(kp[p2][0] * w * scale + offset[0])
        y2 = int(kp[p2][1] * h * scale + offset[1])
        # 绘制骨骼线
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        # 绘制关键点
        cv2.circle(frame, (x1, y1), 4, color, -1)
        cv2.circle(frame, (x2, y2), 4, color, -1)
    return frame

# -------------------------- 主循环 --------------------------
print("\n🎉 实时拳击动作评估系统启动！")
print("🔑 快捷键：")
print("   1/2/3/4 - 切换动作模板（刺拳/直拳/勾拳/摆拳）")
print("   空格     - 暂停/继续")
print("   q/ESC    - 退出程序")
print("="*60)

pause = False
while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret:
            print("❌ 摄像头读取失败")
            break
        frame = cv2.flip(frame, 1)  # 镜像翻转，更自然
        frame_copy = frame.copy()

        # 1. YOLOv8提取人体关键点
        results = pose_model(frame, verbose=False)
        keypoints = None
        if results and len(results[0].keypoints.data) > 0:
            # 取第一个人的关键点（(17,3) → (17,2)，去掉置信度）
            keypoints = results[0].keypoints.data[0][:, :2].cpu().numpy()
            # 绘制用户骨骼（红色）
            frame = draw_skeleton(frame, keypoints, (0, 0, 255))

            # 2. 缓存帧，用于特征提取和相似度计算
            if keypoints is not None:
                frame_buffer.append(keypoints)
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)

                # 3. 动作分类（缓存满10帧才预测）
                if len(frame_buffer) == buffer_size:
                    pred_action = predict_action(frame_buffer)

                # 4. 相似度计算（和当前标准模板对比）
                if current_action_id in standard_actions and len(frame_buffer) >= 5:
                    standard_seq = standard_actions[current_action_id]
                    similarity_score = calculate_similarity(frame_buffer, standard_seq)

                # 5. 绘制标准动作模板（绿色，偏移显示）
                if current_action_id in standard_actions:
                    # 取标准模板的当前帧（循环播放）
                    standard_kp = standard_actions[current_action_id][len(frame_buffer)-1 % len(standard_actions[current_action_id])]
                    # 偏移显示（右边）
                    frame = draw_skeleton(frame, standard_kp, (0, 255, 0), scale=0.8, offset=(30, 0))

        # 6. 绘制UI信息
        # 动作识别结果
        cv2.putText(frame, f"动作识别：{pred_action}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # 相似度打分
        cv2.putText(frame, f"相似度：{similarity_score}%", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # 当前模板
        cv2.putText(frame, f"当前模板：{ACTION_MAP[current_action_id]}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        # 快捷键提示
        cv2.putText(frame, "快捷键：1-刺拳 2-直拳 3-勾拳 4-摆拳 | 空格-暂停 | q-退出", 
                    (20, WINDOW_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # 显示画面
        cv2.imshow("🥊 拳击动作实时评估系统", frame)

    # 键盘事件处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q/ESC退出
        break
    elif key == ord(' '):  # 空格暂停
        pause = not pause
    elif key == ord('1'):  # 1-刺拳
        current_action_id = 0
        frame_buffer = []
    elif key == ord('2'):  # 2-直拳
        current_action_id = 1
        frame_buffer = []
    elif key == ord('3'):  # 3-勾拳
        current_action_id = 2
        frame_buffer = []
    elif key == ord('4'):  # 4-摆拳
        current_action_id = 3
        frame_buffer = []

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("\n👋 程序已退出")