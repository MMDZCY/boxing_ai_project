import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 解决中文乱码核心代码 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
# -------------------------------------------------------------------------

# 加载Day1提取的用户关键点数据
keypoints_seq = np.load("../output/user_boxing_keypoints.npy")

# YOLOv8 Pose 17个关键点固定索引
KEYPOINTS = {
    "nose":0, "left_shoulder":5, "right_shoulder":6,
    "left_elbow":7, "right_elbow":8, "left_wrist":9, "right_wrist":10,
    "left_hip":11, "right_hip":12, "left_knee":13, "right_knee":14,
    "left_ankle":15, "right_ankle":16
}

# -------------------------- 核心工具函数 --------------------------
# 计算三点构成的关节角度
def calculate_joint_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# -------------------------- 逐帧计算核心指标 --------------------------
elbow_angle_seq = []       # 右肘关节角度序列
hip_twist_angle_seq = []   # 肩髋扭转角度序列
center_height_seq = []     # 重心高度序列

for frame_kp in keypoints_seq:
    # 计算右肘关节角度
    shoulder = frame_kp[KEYPOINTS["right_shoulder"]]
    elbow = frame_kp[KEYPOINTS["right_elbow"]]
    wrist = frame_kp[KEYPOINTS["right_wrist"]]
    elbow_angle = calculate_joint_angle(shoulder, elbow, wrist)
    elbow_angle_seq.append(elbow_angle)

    # 计算肩髋扭转角度
    left_shoulder = frame_kp[KEYPOINTS["left_shoulder"]]
    right_shoulder = frame_kp[KEYPOINTS["right_shoulder"]]
    left_hip = frame_kp[KEYPOINTS["left_hip"]]
    right_hip = frame_kp[KEYPOINTS["right_hip"]]
    shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
    hip_vec = right_hip[:2] - left_hip[:2]
    twist_angle = calculate_joint_angle(shoulder_vec, np.array([0,0]), hip_vec)
    hip_twist_angle_seq.append(twist_angle)

    # 计算重心高度（左右髋部平均高度）
    center_height = (left_hip[1] + right_hip[1]) / 2
    center_height_seq.append(center_height)

# -------------------------- 动作量化打分（专业拳击标准） --------------------------
max_elbow_angle = max(elbow_angle_seq)
max_twist_angle = max(hip_twist_angle_seq)
center_fluctuation = np.std(center_height_seq) / np.mean(center_height_seq) * 100

# 100分制权重分配
elbow_score = min(max_elbow_angle / 170 * 40, 40)    # 出拳伸展度40分
twist_score = min(max_twist_angle / 30 * 40, 40)    # 转髋发力40分
stability_score = max(20 - (center_fluctuation - 5) * 2, 0) if center_fluctuation >5 else 20  # 稳定性20分
total_score = elbow_score + twist_score + stability_score

# -------------------------- 控制台输出评估报告 --------------------------
print("="*30 + "拳击动作评估报告" + "="*30)
print(f"📊 肘关节最大伸展角度：{max_elbow_angle:.1f}° | 得分：{elbow_score:.1f}/40")
print(f"📊 肩髋最大扭转角度：{max_twist_angle:.1f}° | 得分：{twist_score:.1f}/40")
print(f"📊 重心波动幅度：{center_fluctuation:.1f}% | 得分：{stability_score:.1f}/20")
print(f"🏆 动作总得分：{total_score:.1f}/100")
print("-"*70)
print("💡 优化建议：")
if max_elbow_angle < 150:
    print("❌ 出拳未完全伸直，肘关节伸展不足，导致发力不充分，击打距离缩短")
if max_twist_angle < 20:
    print("❌ 转髋幅度不足，未用到核心与下肢发力，仅靠手臂出拳，发力效率低且易伤肩")
if center_fluctuation > 15:
    print("❌ 重心波动过大，动作稳定性不足，出拳后易失去平衡，攻防漏洞大")
if total_score >= 80:
    print("✅ 动作标准，发力逻辑正确，保持现有动作框架即可")
elif total_score >= 60:
    print("⚠️ 动作基本合格，重点优化上述提示的问题点即可")
else:
    print("⚠️ 动作存在较多问题，建议先对照标准动作模板纠正基础框架")

# -------------------------- 保存结果 --------------------------
# 保存评估报告
with open("../output/action_evaluate_report.txt", "w", encoding="utf-8") as f:
    f.write(f"动作总得分：{total_score:.1f}/100\n")
    f.write(f"肘关节最大角度：{max_elbow_angle:.1f}°\n")
    f.write(f"肩髋最大扭转角度：{max_twist_angle:.1f}°\n")
    f.write(f"重心波动幅度：{center_fluctuation:.1f}%\n")
# 保存指标时序数据
np.save("../output/action_indicator_seq.npy", {
    "elbow_angle": elbow_angle_seq,
    "twist_angle": hip_twist_angle_seq,
    "center_height": center_height_seq
})

# -------------------------- 绘制正常显示中文的曲线 --------------------------
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(elbow_angle_seq, label="右肘关节角度", color="red")
plt.axhline(y=170, color="gray", linestyle="--", label="专业标准值")
plt.legend()
plt.title("拳击动作核心指标时序变化")

plt.subplot(3,1,2)
plt.plot(hip_twist_angle_seq, label="肩髋扭转角度", color="blue")
plt.axhline(y=30, color="gray", linestyle="--", label="专业标准值")
plt.legend()

plt.subplot(3,1,3)
plt.plot(center_height_seq, label="重心高度", color="green")
plt.legend()
plt.xlabel("视频帧")
plt.tight_layout()
plt.savefig("../output/action_indicator_plot.png", dpi=300)
plt.show()