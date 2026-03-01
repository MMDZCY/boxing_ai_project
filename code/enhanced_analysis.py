import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# -------------------------- 配置 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TARGET_ACTION = "straight"
USER_KP_PATH = "../output/final_result/user_keypoints.npy"
STD_KP_PATH = f"../standard_action/standard_{TARGET_ACTION}.npy"
OUTPUT_DIR = "../output/final_result"

# YOLOv8 17个关键点的连接关系（用于画图）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 脸
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 手臂
    (5, 11), (6, 12), (11, 12), # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16) # 腿
]

# -------------------------- 1. 核心工具函数 --------------------------
def smooth_data(data_seq, window_length=11, polyorder=3):
    """
    使用 Savitzky-Golay 滤波器平滑关键点序列
    解决 YOLOv8 检测时的轻微抖动
    """
    smoothed_seq = np.copy(data_seq)
    # 对 x, y 坐标分别平滑
    for i in range(data_seq.shape[1]):
        smoothed_seq[:, i, 0] = savgol_filter(data_seq[:, i, 0], window_length, polyorder)
        smoothed_seq[:, i, 1] = savgol_filter(data_seq[:, i, 1], window_length, polyorder)
    return smoothed_seq

def calculate_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def simple_dtw_alignment(user_feat, std_feat):
    """简易 DTW 对齐，只返回索引映射"""
    from dtw import dtw
    alignment = dtw(user_feat, std_feat, keep_internals=True)
    return alignment.index1, alignment.index2

# -------------------------- 2. 加载并预处理数据 --------------------------
print("[1/3] 正在加载数据并应用平滑滤波...")

if not os.path.exists(USER_KP_PATH) or not os.path.exists(STD_KP_PATH):
    raise FileNotFoundError("请先运行 boxing_ai_main.py 生成 user_keypoints.npy")

user_kp_raw = np.load(USER_KP_PATH)
std_kp_raw = np.load(STD_KP_PATH)

# 应用平滑滤波 (窗口大小11，多项式阶数3)
user_kp = smooth_data(user_kp_raw)
std_kp = smooth_data(std_kp_raw)

print(f"✅ 数据加载完成：")
print(f"   - 用户动作：{len(user_kp)} 帧 (已平滑)")
print(f"   - 标准动作：{len(std_kp)} 帧 (已平滑)")

# -------------------------- 3. 基于 DTW 找到对应帧 --------------------------
print("\n[2/3] 正在执行时序对齐，定位偏差最大帧...")

# 提取肘关节角度作为对齐特征
def get_elbow_feat(kp_seq):
    feats = []
    for kp in kp_seq:
        ang = calculate_angle(kp[5], kp[7], kp[9])
        feats.append([ang])
    return np.array(feats)

user_feat = get_elbow_feat(user_kp)
std_feat = get_elbow_feat(std_kp)

# 执行对齐
u_idx, s_idx = simple_dtw_alignment(user_feat, std_feat)

# 计算逐帧偏差，找到偏差最大的时刻
core_joints = [5, 6, 7, 8, 9, 10, 11, 12] # 只看核心关节
deviations = []
for ui, si in zip(u_idx, s_idx):
    u_j = user_kp[ui][core_joints][:, :2]
    s_j = std_kp[si][core_joints][:, :2]
    # 归一化：以肩宽为基准，消除拍摄距离的影响
    shoulder_dist_u = np.linalg.norm(user_kp[ui][5] - user_kp[ui][6])
    shoulder_dist_s = np.linalg.norm(std_kp[si][5] - std_kp[si][6])
    avg_shoulder = (shoulder_dist_u + shoulder_dist_s) / 2 + 1e-6
    
    dist = np.mean(np.linalg.norm(u_j - s_j, axis=1)) / avg_shoulder
    deviations.append(dist)

deviations = np.array(deviations)
max_dev_pos = np.argmax(deviations)
target_user_frame = u_idx[max_dev_pos]
target_std_frame = s_idx[max_dev_pos]

print(f"✅ 定位完成：")
print(f"   - 最大偏差出现在：用户第 {target_user_frame} 帧 / 标准第 {target_std_frame} 帧")

# -------------------------- 4. 绘制核心亮点：骨骼叠加对比图 --------------------------
print("\n[3/3] 正在生成骨骼叠加对比图...")

def normalize_and_center(kp):
    """归一化：将人体平移到中心，并根据肩宽缩放"""
    kp = kp.copy()
    # 1. 以右髋为原点平移
    hip_center = (kp[11] + kp[12]) / 2
    kp[:, :2] -= hip_center[:2]
    
    # 2. 垂直翻转 (因为图片坐标系y轴向下)
    kp[:, 1] *= -1
    
    # 3. 根据肩宽缩放
    shoulder_width = np.linalg.norm(kp[5] - kp[6])
    if shoulder_width > 0:
        kp[:, :2] /= shoulder_width
    
    return kp

# 取出目标帧
kp_user_frame = user_kp[target_user_frame]
kp_std_frame = std_kp[target_std_frame]

# 归一化处理（这一步很关键，消除拍摄距离和位置的影响）
kp_user_norm = normalize_and_center(kp_user_frame)
kp_std_norm = normalize_and_center(kp_std_frame)

# 开始画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- 图 1：平滑前后对比 (展示技术细节) ---
ax1.set_title("信号平滑处理对比 (右肘关节角度)", fontsize=14)
user_feat_raw = get_elbow_feat(user_kp_raw)
ax1.plot(user_feat_raw, label="原始数据 (抖动)", color="#ff6b6b", alpha=0.5, linewidth=1)
ax1.plot(user_feat, label="平滑后 (Savitzky-Golay)", color="#ee5a24", linewidth=2.5)
ax1.set_xlabel("视频帧")
ax1.set_ylabel("肘关节角度 (°)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 图 2：骨骼叠加对比 (核心亮点) ---
ax2.set_title(f"动作偏差对比 (用户第 {target_user_frame} 帧)", fontsize=16, fontweight='bold')
ax2.set_aspect('equal')
ax2.axis('off')

# 绘制骨骼函数
def draw_skeleton(ax, kp_norm, color, label, linewidth=3, markersize=6):
    # 绘制连线
    for (i, j) in SKELETON_CONNECTIONS:
        ax.plot([kp_norm[i, 0], kp_norm[j, 0]], 
                [kp_norm[i, 1], kp_norm[j, 1]], 
                color=color, linewidth=linewidth, alpha=0.8, label=label if i == 0 else "")
    # 绘制关键点
    ax.scatter(kp_norm[:, 0], kp_norm[:, 1], color=color, s=markersize**2, zorder=10)

# 绘制：标准动作(绿) + 用户动作(红)
draw_skeleton(ax2, kp_std_norm, "#27ae60", "标准动作 (教练)", linewidth=4, markersize=7)
draw_skeleton(ax2, kp_user_norm, "#e74c3c", "你的动作", linewidth=2, markersize=5)

# 手动添加图例，避免重复
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#27ae60', lw=4),
    Line2D([0], [0], color='#e74c3c', lw=2)
]
ax2.legend(custom_lines, ['标准动作 (模板)', '你的动作'], loc='lower right', fontsize=12)

# 画个箭头指出差异大的地方（比如右拳位置）
wrist_user = kp_user_norm[9]
wrist_std = kp_std_norm[9]
if np.linalg.norm(wrist_user - wrist_std) > 0.1:
    ax2.annotate('出拳位置偏差', 
                 xy=(wrist_user[0], wrist_user[1]), 
                 xytext=(wrist_std[0], wrist_std[1]),
                 arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2, connectionstyle="arc3"),
                 fontsize=12, color='#d35400', fontweight='bold')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "4_Enhanced_Skeleton_Comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 增强分析图已保存至：{output_path}")
plt.show()