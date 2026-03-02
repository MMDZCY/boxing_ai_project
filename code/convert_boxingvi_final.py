import os
import numpy as np
import pandas as pd

# -------------------------- 核心配置（无需修改） --------------------------
# 数据集根路径
DATASET_ROOT = r"D:\boxing_ai_project\boxingvi_dataset"
ANNOTATION_FOLDER = os.path.join(DATASET_ROOT, "Annotation_files")
SKELETON_FOLDER = os.path.join(DATASET_ROOT, "Skeleton_data")

# 动作映射（严格匹配BoxingVI的标注）
ACTION_MAP = {
    "Jab": {
        "name": "jab",
        "project_label": 0,
        "standard_save_path": "../standard_action/standard_jab.npy"
    },
    "Cross": {
        "name": "straight",
        "project_label": 1,
        "standard_save_path": "../standard_action/standard_straight.npy"
    },
    "Lead Hook": {
        "name": "hook",
        "project_label": 2,
        "standard_save_path": "../standard_action/standard_hook.npy"
    },
    "Rear Hook": {
        "name": "swing",
        "project_label": 3,
        "standard_save_path": "../standard_action/standard_swing.npy"
    },
    "cross": {  # 兼容小写标注
        "name": "straight",
        "project_label": 1,
        "standard_save_path": "../standard_action/standard_straight.npy"
    }
}

# 窗口配置
CLASSIFY_WINDOW_SIZE = 10  # 时序窗口大小（10帧）
SUBJECT_RANGE = range(1, 11)  # 处理V1-V10

# 输出配置
OUTPUT_ANGLE_CSV = "../model/action_dataset_boxingvi_angle.csv"    # 角度均值特征（适配随机森林）
OUTPUT_SEQ_CSV = "../model/action_dataset_boxingvi_seq.csv"        # 原始时序特征（适配LSTM）

# -------------------------- 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    """计算三个关键点的夹角（度数）"""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_angle_feature(kp_seq):
    """提取8维角度均值特征（适配随机森林）"""
    window_features = []
    for kp in kp_seq:
        # 计算8个核心角度特征
        left_elbow = calculate_angle(kp[5], kp[7], kp[9])    # 左肘：左肩-左肘-左手腕
        right_elbow = calculate_angle(kp[6], kp[8], kp[10])  # 右肘：右肩-右肘-右手腕
        left_shoulder = calculate_angle(kp[11], kp[5], kp[7])# 左肩：左髋-左肩-左肘
        right_shoulder = calculate_angle(kp[12], kp[6], kp[8])# 右肩：右髋-右肩-右肘
        shoulder_twist = calculate_angle(np.array([0,0]), kp[6][:2]-kp[5][:2], kp[12][:2]-kp[11][:2])# 肩扭转
        left_arm_vertical = calculate_angle(kp[7], kp[9], np.array([kp[9][0], 0]))# 左手臂垂直角度
        right_arm_vertical = calculate_angle(kp[8], kp[10], np.array([kp[10][0], 0]))# 右手臂垂直角度
        wrist_diff = (kp[9][1] - kp[10][1]) / (np.linalg.norm(kp[5]-kp[6]) + 1e-6)# 手腕高度差
        
        window_features.append([
            left_elbow, right_elbow, left_shoulder, right_shoulder,
            shoulder_twist, left_arm_vertical, right_arm_vertical, wrist_diff
        ])
    # 返回窗口内的均值
    return np.mean(window_features, axis=0)

def extract_original_seq_feature(kp_seq):
    """提取原始时序特征（适配LSTM）：10帧×17关键点×2坐标 = 340维"""
    # 展平序列：(10, 17, 2) → (340,)
    return kp_seq.flatten()

def process_skeleton_data(skeleton_data, frame_idx):
    """统一处理不同格式的骨架数据（兼容V6的特殊形状）"""
    frame_data = skeleton_data[frame_idx]
    
    # 处理形状：(25,17,2) 或 (1,17,3)
    if len(frame_data.shape) == 3:
        # 取第0个人的关键点
        person_kp = frame_data[0]
        # 取前两个坐标（x,y），忽略置信度（如果有）
        person_kp = person_kp[:, :2]
    else:
        # 异常数据处理
        person_kp = np.zeros((17, 2))
    
    return person_kp

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("="*60)
    print("🥊 BoxingVI 数据集转换程序（最终完整版）")
    print("="*60)

    # 创建输出文件夹
    os.makedirs("../standard_action", exist_ok=True)
    os.makedirs("../model", exist_ok=True)

    # 初始化变量
    # 1. 收集标准动作模板的关键点
    action_keypoint_collections = {v["name"]: [] for v in ACTION_MAP.values()}
    # 2. 收集训练样本（角度特征+时序特征）
    angle_train_samples = []    # 8维角度均值特征
    seq_train_samples = []      # 340维原始时序特征

    # 遍历所有受试者（V1-V10）
    for subject_id in SUBJECT_RANGE:
        print(f"\n🔄 正在处理受试者 V{subject_id}")
        annotation_path = os.path.join(ANNOTATION_FOLDER, f"V{subject_id}.xlsx")
        skeleton_path = os.path.join(SKELETON_FOLDER, f"V{subject_id}.npy")
        
        # 检查文件是否存在
        if not os.path.exists(annotation_path) or not os.path.exists(skeleton_path):
            print(f"⚠️  跳过V{subject_id}：文件不存在")
            continue

        try:
            # 加载标注和骨架数据
            annotation_df = pd.read_excel(annotation_path)
            full_skeleton_data = np.load(skeleton_path)
            
            # 打印数据形状（调试用）
            print(f"✅ 加载成功：标注 {len(annotation_df)} 行，骨架形状 {full_skeleton_data.shape}")
            
            # 取最小长度（避免标注和骨架帧数不匹配）
            min_len = min(len(annotation_df), len(full_skeleton_data))
            
            # 遍历每帧数据
            for i in range(min_len):
                try:
                    # 1. 获取标注信息
                    row = annotation_df.iloc[i]
                    class_label = str(row["Class"]).strip()
                    
                    # 跳过非目标动作
                    if class_label not in ACTION_MAP:
                        continue
                    
                    # 2. 获取当前帧的关键点（统一格式）
                    kp = process_skeleton_data(full_skeleton_data, i)
                    
                    # 3. 获取动作配置
                    action_config = ACTION_MAP[class_label]
                    action_name = action_config["name"]
                    project_label = action_config["project_label"]

                    # 4. 收集关键点用于生成标准模板
                    action_keypoint_collections[action_name].append(kp)

                    # 5. 生成时序窗口样本（滑动窗口）
                    start_idx = max(0, i - CLASSIFY_WINDOW_SIZE // 2)
                    end_idx = min(len(full_skeleton_data), start_idx + CLASSIFY_WINDOW_SIZE)
                    
                    # 补全窗口（避免帧数不足）
                    if end_idx - start_idx < CLASSIFY_WINDOW_SIZE:
                        start_idx = max(0, end_idx - CLASSIFY_WINDOW_SIZE)
                    
                    # 提取窗口内的所有帧关键点
                    window_kp_list = []
                    for j in range(start_idx, end_idx):
                        window_kp = process_skeleton_data(full_skeleton_data, j)
                        window_kp_list.append(window_kp)
                    
                    # 确保窗口大小正确
                    if len(window_kp_list) == CLASSIFY_WINDOW_SIZE:
                        # 5.1 提取角度均值特征（适配随机森林）
                        angle_feature = extract_angle_feature(window_kp_list)
                        angle_train_samples.append(np.append(angle_feature, project_label))
                        
                        # 5.2 提取原始时序特征（适配LSTM）
                        seq_feature = extract_original_seq_feature(np.array(window_kp_list))
                        seq_train_samples.append(np.append(seq_feature, project_label))
                        
                except Exception as e:
                    # 跳过异常帧，不中断整体流程
                    continue

        except Exception as e:
            print(f"❌ V{subject_id} 处理失败：{str(e)[:50]}")
            continue

    # -------------------------- 生成标准动作模板 --------------------------
    print("\n📦 正在生成标准动作模板...")
    for action_name, kp_list in action_keypoint_collections.items():
        if len(kp_list) == 0:
            print(f"⚠️  跳过 {action_name}：无有效数据")
            continue
        
        # 取前200帧作为标准模板（保证长度统一）
        template_seq = np.array(kp_list[:200])
        # 获取保存路径
        save_path = [v["standard_save_path"] for v in ACTION_MAP.values() if v["name"] == action_name][0]
        # 保存模板
        np.save(save_path, template_seq)
        print(f"✅ 生成 {action_name} 标准模板：{len(template_seq)} 帧，保存至 {save_path}")

    # -------------------------- 保存训练集 --------------------------
    # 1. 保存角度均值特征训练集（适配随机森林）
    if len(angle_train_samples) > 0:
        angle_df = pd.DataFrame(angle_train_samples, columns=[
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "shoulder_twist", "left_arm_vertical", "right_arm_vertical", "wrist_diff", "label"
        ])
        angle_df.to_csv(OUTPUT_ANGLE_CSV, index=False)
        print(f"\n✅ 角度特征训练集生成完成：{len(angle_train_samples)} 个样本，保存至 {OUTPUT_ANGLE_CSV}")
        # 打印样本分布
        print("📊 角度特征样本分布：")
        label_count = angle_df['label'].value_counts().sort_index()
        for label, count in label_count.items():
            action_name = [v["name"] for v in ACTION_MAP.values() if v["project_label"] == label][0]
            print(f"   {action_name}：{count} 个样本")
    else:
        print("\n❌ 未生成角度特征训练集：无有效样本")

    # 2. 保存原始时序特征训练集（适配LSTM）
    if len(seq_train_samples) > 0:
        # 生成340维特征的列名
        seq_columns = [f"kp_{i}" for i in range(340)] + ["label"]
        seq_df = pd.DataFrame(seq_train_samples, columns=seq_columns)
        seq_df.to_csv(OUTPUT_SEQ_CSV, index=False)
        print(f"\n✅ 时序特征训练集生成完成：{len(seq_train_samples)} 个样本，保存至 {OUTPUT_SEQ_CSV}")
    else:
        print("\n❌ 未生成时序特征训练集：无有效样本")

    print("\n="*60)
    print("🎉 数据集转换完成！所有文件已生成！")
    print("="*60)