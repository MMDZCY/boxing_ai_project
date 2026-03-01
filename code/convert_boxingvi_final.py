import os
import numpy as np
import pandas as pd

# -------------------------- 配置 --------------------------
DATASET_ROOT = r"D:\boxing_ai_project\boxingvi_dataset"
ANNOTATION_FOLDER = os.path.join(DATASET_ROOT, "Annotation_files")
SKELETON_FOLDER = os.path.join(DATASET_ROOT, "Skeleton_data")

# 动作映射
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
    "cross": {
        "name": "straight",
        "project_label": 1,
        "standard_save_path": "../standard_action/standard_straight.npy"
    }
}

CLASSIFY_WINDOW_SIZE = 10
SUBJECT_RANGE = range(1, 11)

# -------------------------- 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_classify_feature(kp_seq):
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

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("="*60)
    print("🥊 BoxingVI 数据集转换程序（完美适配数据结构）")
    print("="*60)

    if not os.path.exists(ANNOTATION_FOLDER) or not os.path.exists(SKELETON_FOLDER):
        raise FileNotFoundError("请检查数据集路径")

    os.makedirs("../standard_action", exist_ok=True)
    os.makedirs("../model", exist_ok=True)

    all_train_samples = []
    action_keypoint_collections = {v["name"]: [] for v in ACTION_MAP.values()}

    for subject_id in SUBJECT_RANGE:
        print(f"\n🔄 正在处理受试者 V{subject_id}")
        annotation_path = os.path.join(ANNOTATION_FOLDER, f"V{subject_id}.xlsx")
        skeleton_path = os.path.join(SKELETON_FOLDER, f"V{subject_id}.npy")
        
        if not os.path.exists(annotation_path) or not os.path.exists(skeleton_path):
            print(f"⚠️  跳过V{subject_id}")
            continue

        try:
            annotation_df = pd.read_excel(annotation_path)
            # 数据结构：(帧数, 25个人, 17个关键点, 2个坐标)
            full_skeleton_data = np.load(skeleton_path)
            print(f"✅ 加载成功：标注 {len(annotation_df)} 行，骨架数据形状 {full_skeleton_data.shape}")
        except Exception as e:
            print(f"❌ V{subject_id} 加载失败：{e}")
            continue

        # 核心逻辑：第i行标注 对应 第i帧数据
        min_len = min(len(annotation_df), len(full_skeleton_data))
        for i in range(min_len):
            try:
                row = annotation_df.iloc[i]
                # 关键修改：取第0个人的关键点（主要跟踪目标）
                # 数据形状：(25, 17, 2) -> 取 [0, :, :] -> (17, 2)
                kp = full_skeleton_data[i][0]
                
                class_label = str(row["Class"]).strip()
                if class_label not in ACTION_MAP:
                    continue
                
                action_config = ACTION_MAP[class_label]
                action_name = action_config["name"]
                project_label = action_config["project_label"]

                # 1. 收集关键点，用于生成标准模板
                action_keypoint_collections[action_name].append(kp)

                # 2. 生成训练样本（滑动窗口）
                start_idx = max(0, i - CLASSIFY_WINDOW_SIZE // 2)
                end_idx = min(len(full_skeleton_data), start_idx + CLASSIFY_WINDOW_SIZE)
                if end_idx - start_idx < CLASSIFY_WINDOW_SIZE:
                    start_idx = max(0, end_idx - CLASSIFY_WINDOW_SIZE)
                
                # 提取窗口内的关键点（都取第0个人）
                window_kp_list = []
                for j in range(start_idx, end_idx):
                    window_kp_list.append(full_skeleton_data[j][0])
                
                if len(window_kp_list) == CLASSIFY_WINDOW_SIZE:
                    feature = extract_classify_feature(window_kp_list)
                    all_train_samples.append(np.append(feature, project_label))
            except Exception as e:
                continue

    # 生成标准动作模板
    print("\n📦 正在生成标准动作模板...")
    for action_name, kp_list in action_keypoint_collections.items():
        if len(kp_list) < 50:
            print(f"⚠️  跳过 {action_name}：数据不足")
            continue
        # 取前200帧作为标准模板
        template_seq = np.array(kp_list[:200])
        save_path = [v["standard_save_path"] for v in ACTION_MAP.values() if v["name"] == action_name][0]
        np.save(save_path, template_seq)
        print(f"✅ 生成 {action_name} 标准模板：{len(template_seq)} 帧")

    # 保存训练集
    if len(all_train_samples) > 0:
        df = pd.DataFrame(all_train_samples, columns=[
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "shoulder_twist", "left_arm_vertical", "right_arm_vertical", "wrist_diff", "label"
        ])
        save_path = "../model/action_dataset_boxingvi.csv"
        df.to_csv(save_path, index=False)
        print(f"\n✅ 训练集生成完成！共 {len(all_train_samples)} 个专业样本")
        print("\n📊 样本分布：")
        label_count = df['label'].value_counts().sort_index()
        for label, count in label_count.items():
            action_name = [v["name"] for v in ACTION_MAP.values() if v["project_label"] == label][0]
            print(f"   {action_name}：{count} 个样本")
    else:
        print("\n❌ 未提取到有效样本")

    print("="*60)