import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
import os

# -------------------------- 全局配置 --------------------------
TARGET_ACTION_TYPE = "straight" 
USER_KEYPOINTS_PATH = "../output/user_boxing_keypoints.npy"
STANDARD_KEYPOINTS_PATH = f"../standard_action/standard_{TARGET_ACTION_TYPE}.npy"
OUTPUT_PLOT_PATH = "../output/alignment_result.png"

# -------------------------- 0. 检查文件并加载 --------------------------
if not os.path.exists(USER_KEYPOINTS_PATH) or not os.path.exists(STANDARD_KEYPOINTS_PATH):
    raise FileNotFoundError("请确保用户关键点和标准动作模板文件存在")

print(f"🚀 开始动作对齐：用户动作 vs 标准{TARGET_ACTION_TYPE}拳")
user_kp_seq = np.load(USER_KEYPOINTS_PATH)
standard_kp_seq = np.load(STANDARD_KEYPOINTS_PATH)
print(f"✅ 数据加载成功：用户 {len(user_kp_seq)} 帧，标准 {len(standard_kp_seq)} 帧")

# YOLOv8 关键点索引
KEYPOINTS = {
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8, 
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12
}

# -------------------------- 1. 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_feature_sequence(kp_seq):
    """提取特征：这里我们只用【右肘关节角度】这一个核心特征来做对齐和展示"""
    feature_seq = []
    for kp in kp_seq:
        elbow_angle = calculate_angle(
            kp[KEYPOINTS["right_shoulder"]],
            kp[KEYPOINTS["right_elbow"]],
            kp[KEYPOINTS["right_wrist"]]
        )
        feature_seq.append([elbow_angle]) # 注意：这里变成了单变量，方便绘图
    return np.array(feature_seq)

# -------------------------- 2. 执行 DTW 对齐 --------------------------
print("🔄 正在执行时序对齐...")
# 为了绘图，我们这次只用【右肘关节角度】做对齐
user_feature = extract_feature_sequence(user_kp_seq)
standard_feature = extract_feature_sequence(standard_kp_seq)

alignment = dtw(user_feature, standard_feature, keep_internals=True)
user_aligned_idx = alignment.index1
standard_aligned_idx = alignment.index2

# -------------------------- 3. 偏差分析 --------------------------
print("📊 正在计算动作偏差...")
frame_deviations = []
core_joint_indices = [5, 6, 7, 8, 9, 10, 11, 12]

for u_idx, s_idx in zip(user_aligned_idx, standard_aligned_idx):
    u_joints = user_kp_seq[u_idx][core_joint_indices][:, :2]
    s_joints = standard_kp_seq[s_idx][core_joint_indices][:, :2]
    dist = np.mean(np.linalg.norm(u_joints - s_joints, axis=1))
    frame_deviations.append(dist)

frame_deviations = np.array(frame_deviations)
max_dev_idx = np.argmax(frame_deviations)
problem_user_frame = user_aligned_idx[max_dev_idx]

# -------------------------- 4. 生成报告 --------------------------
print("\n" + "="*50)
print("📋 拳击动作偏差分析报告")
print("="*50)
print(f"对比动作：标准 {TARGET_ACTION_TYPE} 拳")
print(f"DTW 对齐距离：{alignment.distance:.2f}")
print(f"动作整体平均偏差：{np.mean(frame_deviations):.2f} 像素")
print(f"⚠️  最大偏差出现在：用户动作第 {problem_user_frame} 帧")
print(f"⚠️  最大偏差值：{frame_deviations[max_dev_idx]:.2f} 像素")
print("\n💡 核心建议：")
print(f"1. 你的动作整体偏差较大，建议先练习【蹬地转髋】的基础发力逻辑")
print(f"2. 重点查看视频第 {problem_user_frame} 帧，这是你动作变形最严重的时刻")
print("="*50)

# -------------------------- 5. 修复后的可视化 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 图1：对齐后的肘关节角度对比（核心修改：去掉了twoway，改为手动绘制对齐曲线）
ax1.set_title("动作对齐对比：右肘关节角度变化", fontsize=14)
# 绘制原始曲线
ax1.plot(user_feature, label="你的动作", color="#ff6b6b", alpha=0.7, linewidth=2)
ax1.plot(standard_feature, label="标准动作", color="#4ecdc4", alpha=0.7, linewidth=2)
# 绘制对齐连线（每隔20帧画一条，避免太乱）
for i in range(0, len(user_aligned_idx), 20):
    u_idx = user_aligned_idx[i]
    s_idx = standard_aligned_idx[i]
    ax1.plot([u_idx, s_idx], [user_feature[u_idx], standard_feature[s_idx]], 
             color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax1.set_ylabel("肘关节角度 (°)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：偏差曲线
ax2.set_title("动作全流程偏差曲线", fontsize=14)
ax2.plot(frame_deviations, label="帧偏差值", color="#ff4757", linewidth=2)
ax2.axvline(x=max_dev_idx, color="#2f3542", linestyle="--", label="最大偏差点")
ax2.fill_between(range(len(frame_deviations)), frame_deviations, alpha=0.3, color="#ff4757")
ax2.set_xlabel("对齐后帧序号")
ax2.set_ylabel("核心关节平均偏差 (像素)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
print(f"\n🖼️  可视化结果已保存至：{OUTPUT_PLOT_PATH}")
plt.show()