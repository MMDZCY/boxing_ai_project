import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from dtw import dtw

# -------------------------- 全局配置与初始化 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 仅需修改这里！！！
INPUT_VIDEO_PATH = "../input_video/test_boxing.mp4"  # 你的测试视频路径
TARGET_ACTION_TYPE = "straight"                       # 对比的标准动作: jab/straight/hook

# 自动路径配置
BASE_OUTPUT_DIR = "../output/final_result"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

STANDARD_ACTION_PATH = f"../standard_action/standard_{TARGET_ACTION_TYPE}.npy"
MODEL = YOLO('yolov8n-pose.pt')

# YOLOv8 关键点索引
KEYPOINTS = {
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12
}

print("="*60)
print("🏆  拳击动作 AI 优化系统 - 全链路一键分析")
print("="*60)

# -------------------------- 模块 1：姿态提取 --------------------------
def run_pose_extraction(video_path):
    print("\n[1/4] 正在进行人体姿态提取...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(BASE_OUTPUT_DIR, "1_pose_annotated.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    all_keypoints = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 进度条提示
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"    处理进度: {frame_count}/{total_frames} 帧", end='\r')

        results = MODEL(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        if results[0].keypoints.data.shape[0] > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            all_keypoints.append(keypoints)
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    
    kp_path = os.path.join(BASE_OUTPUT_DIR, "user_keypoints.npy")
    np.save(kp_path, np.array(all_keypoints))
    
    print(f"    ✅ 姿态提取完成！")
    print(f"    📁 标注视频：{out_video_path}")
    return np.array(all_keypoints)

# -------------------------- 模块 2：动作量化评估 --------------------------
def calculate_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def run_action_evaluation(keypoints_seq):
    print("\n[2/4] 正在进行动作量化评估...")
    
    elbow_angle_seq = []
    hip_twist_angle_seq = []
    center_height_seq = []

    for frame_kp in keypoints_seq:
        # 1. 右肘关节角度
        elbow_angle = calculate_angle(
            frame_kp[KEYPOINTS["right_shoulder"]],
            frame_kp[KEYPOINTS["right_elbow"]],
            frame_kp[KEYPOINTS["right_wrist"]]
        )
        elbow_angle_seq.append(elbow_angle)

        # 2. 肩髋扭转角度
        shoulder_vec = frame_kp[KEYPOINTS["right_shoulder"]][:2] - frame_kp[KEYPOINTS["left_shoulder"]][:2]
        hip_vec = frame_kp[KEYPOINTS["right_hip"]][:2] - frame_kp[KEYPOINTS["left_hip"]][:2]
        twist_angle = calculate_angle(np.array([0,0]), shoulder_vec, hip_vec)
        hip_twist_angle_seq.append(twist_angle)

        # 3. 重心高度
        center_height = (frame_kp[KEYPOINTS["left_hip"]][1] + frame_kp[KEYPOINTS["right_hip"]][1]) / 2
        center_height_seq.append(center_height)

    # 打分逻辑
    max_elbow = max(elbow_angle_seq)
    max_twist = max(hip_twist_angle_seq)
    center_fluct = np.std(center_height_seq) / np.mean(center_height_seq) * 100

    elbow_score = min(max_elbow / 170 * 40, 40)
    twist_score = min(max_twist / 30 * 40, 40)
    stability_score = max(20 - (center_fluct - 5) * 2, 0) if center_fluct >5 else 20
    total_score = elbow_score + twist_score + stability_score

    # 生成报告
    suggestions = []
    if max_elbow < 150: suggestions.append("❌ 出拳未完全伸直，发力不充分")
    if max_twist < 20: suggestions.append("❌ 转髋幅度不足，仅靠手臂发力")
    if center_fluct > 15: suggestions.append("❌ 重心波动过大，下盘不稳")
    if not suggestions: suggestions.append("✅ 动作框架良好，保持即可")

    # 保存报告
    report_path = os.path.join(BASE_OUTPUT_DIR, "2_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*30 + " 拳击动作评估报告 " + "="*30 + "\n")
        f.write(f"🏆 总得分：{total_score:.1f}/100\n")
        f.write(f"📊 肘关节最大角度：{max_elbow:.1f}° | 得分：{elbow_score:.1f}/40\n")
        f.write(f"📊 肩髋最大扭转：{max_twist:.1f}° | 得分：{twist_score:.1f}/40\n")
        f.write(f"📊 重心波动幅度：{center_fluct:.1f}% | 得分：{stability_score:.1f}/20\n")
        f.write("\n优化建议：\n")
        for s in suggestions: f.write(f"- {s}\n")

    # 绘制曲线
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(elbow_angle_seq, color="#ff6b6b", label="右肘关节角度")
    axes[0].axhline(170, color="gray", linestyle="--", label="标准值")
    axes[0].set_title("动作核心指标时序变化")
    axes[0].legend()
    
    axes[1].plot(hip_twist_angle_seq, color="#4ecdc4", label="肩髋扭转角度")
    axes[1].axhline(30, color="gray", linestyle="--", label="标准值")
    axes[1].legend()
    
    axes[2].plot(center_height_seq, color="#45b7d1", label="重心高度")
    axes[2].legend()
    axes[2].set_xlabel("视频帧")
    
    plt.tight_layout()
    plot_path = os.path.join(BASE_OUTPUT_DIR, "2_evaluation_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"    ✅ 动作评估完成！")
    print(f"    🏆 总得分：{total_score:.1f}/100")
    return total_score, suggestions

# -------------------------- 模块 3：动作对齐与偏差定位 --------------------------
def run_action_alignment(user_kp_seq):
    print("\n[3/4] 正在进行动作对齐与偏差定位...")
    
    if not os.path.exists(STANDARD_ACTION_PATH):
        print(f"    ⚠️  未找到标准动作模板，跳过对齐步骤")
        return None

    standard_kp_seq = np.load(STANDARD_ACTION_PATH)

    # 特征提取（只用肘关节角度）
    def get_feat(seq):
        feats = []
        for kp in seq:
            ang = calculate_angle(kp[5], kp[7], kp[9])
            feats.append([ang])
        return np.array(feats)

    user_feat = get_feat(user_kp_seq)
    std_feat = get_feat(standard_kp_seq)

    # DTW 对齐
    alignment = dtw(user_feat, std_feat, keep_internals=True)
    u_idx_aligned = alignment.index1
    s_idx_aligned = alignment.index2

    # 计算偏差
    frame_devs = []
    core_joints = [5,6,7,8,9,10,11,12]
    for u_i, s_i in zip(u_idx_aligned, s_idx_aligned):
        u_j = user_kp_seq[u_i][core_joints][:, :2]
        s_j = standard_kp_seq[s_i][core_joints][:, :2]
        dist = np.mean(np.linalg.norm(u_j - s_j, axis=1))
        frame_devs.append(dist)
    frame_devs = np.array(frame_devs)
    max_dev_idx = np.argmax(frame_devs)
    problem_frame = u_idx_aligned[max_dev_idx]

    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.set_title("动作对齐对比：肘关节角度")
    ax1.plot(user_feat, label="你的动作", color="#ff6b6b", alpha=0.7)
    ax1.plot(std_feat, label="标准动作", color="#4ecdc4", alpha=0.7)
    # 稀疏绘制对齐线
    for i in range(0, len(u_idx_aligned), 30):
        ax1.plot([u_idx_aligned[i], s_idx_aligned[i]], 
                 [user_feat[u_idx_aligned[i]], std_feat[s_idx_aligned[i]]], 
                 color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.legend()

    ax2.set_title("动作偏差曲线")
    ax2.plot(frame_devs, color="#ff4757", label="帧偏差")
    ax2.axvline(max_dev_idx, color="k", linestyle="--", label="最大偏差点")
    ax2.fill_between(range(len(frame_devs)), frame_devs, alpha=0.3, color="#ff4757")
    ax2.legend()
    ax2.set_xlabel("对齐后帧序号")

    plt.tight_layout()
    plot_path = os.path.join(BASE_OUTPUT_DIR, "3_alignment_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"    ✅ 偏差定位完成！")
    print(f"    ⚠️  最大偏差在第 {problem_frame} 帧")
    return problem_frame

# -------------------------- 主流程入口 --------------------------
if __name__ == "__main__":
    try:
        # 1. 姿态提取
        user_kp = run_pose_extraction(INPUT_VIDEO_PATH)
        
        # 2. 动作评估
        score, suggs = run_action_evaluation(user_kp)
        
        # 3. 动作对齐
        problem_frame = run_action_alignment(user_kp)

        # 4. 最终总结
        print("\n" + "="*60)
        print("🎉  全流程分析完成！")
        print("="*60)
        print(f"📁 所有结果已保存至：{os.path.abspath(BASE_OUTPUT_DIR)}")
        print("\n📋 核心总结：")
        print(f"   1. 动作总得分：{score:.1f}/100")
        print(f"   2. 主要问题：{suggs[0] if suggs else '无'}")
        if problem_frame:
            print(f"   3. 重点纠正帧：第 {problem_frame} 帧")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序出错：{e}")
        import traceback
        traceback.print_exc()