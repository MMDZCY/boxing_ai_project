# 文件路径: d:\boxing_ai_project\code\data_augmentation.py
import numpy as np
import pandas as pd
import os
from collections import deque

class BoxingDataAugmentor:
    """拳击动作数据增强器"""
    
    def __init__(self, noise_std=0.02, max_shift=5, max_scale=0.1, max_rotate=10):
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.max_rotate = max_rotate
    
    def add_gaussian_noise(self, keypoints):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, keypoints.shape)
        return keypoints + noise
    
    def shift_keypoints(self, keypoints):
        """平移关键点"""
        dx = np.random.uniform(-self.max_shift, self.max_shift)
        dy = np.random.uniform(-self.max_shift, self.max_shift)
        shifted = keypoints.copy()
        shifted[:, 0] += dx
        shifted[:, 1] += dy
        return shifted
    
    def scale_keypoints(self, keypoints):
        """缩放关键点"""
        scale_x = 1 + np.random.uniform(-self.max_scale, self.max_scale)
        scale_y = 1 + np.random.uniform(-self.max_scale, self.max_scale)
        
        # 找到中心点
        center = np.mean(keypoints[:, :2], axis=0)
        
        scaled = keypoints.copy()
        scaled[:, 0] = center[0] + (keypoints[:, 0] - center[0]) * scale_x
        scaled[:, 1] = center[1] + (keypoints[:, 1] - center[1]) * scale_y
        
        return scaled
    
    def rotate_keypoints(self, keypoints):
        """旋转关键点"""
        angle = np.random.uniform(-self.max_rotate, self.max_rotate)
        angle_rad = np.radians(angle)
        
        # 找到中心点
        center = np.mean(keypoints[:, :2], axis=0)
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        rotated = keypoints.copy()
        for i in range(len(keypoints)):
            # 平移到原点
            x = keypoints[i, 0] - center[0]
            y = keypoints[i, 1] - center[1]
            # 旋转
            new_x, new_y = rot_matrix @ np.array([x, y])
            # 平移回去
            rotated[i, 0] = new_x + center[0]
            rotated[i, 1] = new_y + center[1]
        
        return rotated
    
    def time_warp(self, keypoint_sequence, warp_factor=0.2):
        """时间扭曲 - 改变动作速度"""
        n_frames = len(keypoint_sequence)
        if n_frames < 3:
            return keypoint_sequence
        
        # 生成新的时间索引
        warp_amount = int(n_frames * warp_factor)
        new_indices = np.linspace(0, n_frames-1, n_frames + np.random.randint(-warp_amount, warp_amount+1))
        new_indices = np.clip(new_indices, 0, n_frames-1).astype(int)
        
        # 插值
        warped = []
        for idx in new_indices:
            warped.append(keypoint_sequence[idx])
        
        return np.array(warped)
    
    def mirror_keypoints(self, keypoints):
        """镜像翻转（左右互换）"""
        mirrored = keypoints.copy()
        
        # 左右关键点互换
        swap_pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for i, j in swap_pairs:
            if i < len(keypoints) and j < len(keypoints):
                mirrored[[i, j]] = mirrored[[j, i]]
        
        # X轴翻转
        mirrored[:, 0] = -mirrored[:, 0]
        
        return mirrored
    
    def augment_sequence(self, keypoint_sequence, label, num_augments=3):
        """增强一个动作序列"""
        augmented_data = [(keypoint_sequence, label)]  # 原始数据
        
        for _ in range(num_augments):
            aug_seq = keypoint_sequence.copy()
            
            # 随机选择增强方法
            methods = [
                self.add_gaussian_noise,
                self.shift_keypoints,
                self.scale_keypoints,
                self.rotate_keypoints,
            ]
            
            # 随机应用2-3种方法
            num_methods = np.random.randint(2, 4)
            selected_methods = np.random.choice(methods, num_methods, replace=False)
            
            for method in selected_methods:
                # 逐帧应用
                aug_seq = np.array([method(frame) for frame in aug_seq])
            
            augmented_data.append((aug_seq, label))
        
        return augmented_data


def augment_boxing_dataset(seq_data_path, angle_data_path, output_dir, augment_factor=3):
    """增强整个BoxingVI数据集"""
    print("="*70)
    print("📊 BoxingVI 数据集增强程序")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n[加载] 正在加载原始数据集...")
    seq_df = pd.read_csv(seq_data_path)
    angle_df = pd.read_csv(angle_data_path)
    
    print(f"   时序特征样本数: {len(seq_df)}")
    print(f"   角度特征样本数: {len(angle_df)}")
    
    # 解析时序数据
    print("\n[解析] 正在解析时序特征...")
    X = seq_df.iloc[:, :-1].values  # 除了最后一列(label)的所有列
    y = seq_df['label'].values.astype(int)
    
    # 重塑为时序数据
    # 340维 = 10帧 × 17关键点 × 2坐标
    n_samples = len(X)
    X_seq = X.reshape(n_samples, 10, 34)  # (样本, 10帧, 34维)
    # 再重塑为 (样本, 10帧, 17, 2) 用于数据增强
    X_seq = X_seq.reshape(n_samples, 10, 17, 2)
    
    X_seq = np.array(X_seq)
    y = np.array(y)
    
    # 创建增强器
    augmentor = BoxingDataAugmentor()
    
    # 数据增强
    print(f"\n[增强] 正在进行数据增强 (增强因子: {augment_factor})...")
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X_seq)):
        seq = X_seq[i]
        label = y[i]
        
        # 增强这个样本
        aug_results = augmentor.augment_sequence(seq, label, num_augments=augment_factor)
        
        for aug_seq, aug_label in aug_results:
            # 重塑回原来的格式
            aug_seq_flat = aug_seq.flatten()
            X_augmented.append(aug_seq_flat)
            y_augmented.append(aug_label)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    # 创建增强后的DataFrame（保持原始格式）
    print("\n[保存] 正在保存增强后的数据集...")
    # 重塑回原始格式 (样本数, 340)
    X_augmented_flat = X_augmented.reshape(len(X_augmented), -1)
    # 创建列名: kp_0, kp_1, ..., kp_339, label
    columns = [f'kp_{i}' for i in range(X_augmented_flat.shape[1])] + ['label']
    # 合并数据
    augmented_data = np.column_stack([X_augmented_flat, y_augmented])
    augmented_seq_df = pd.DataFrame(augmented_data, columns=columns)
    
    # 也增强角度特征（简单复制+小噪声）
    augmented_angle_data = []
    for idx, row in angle_df.iterrows():
        # 原始数据
        augmented_angle_data.append(row.tolist())
        
        # 增强数据
        for _ in range(augment_factor):
            noisy_row = row.tolist()
            # 给特征添加小噪声（除了最后一列label）
            for i in range(len(noisy_row)-1):
                noisy_row[i] += np.random.normal(0, 2)
            augmented_angle_data.append(noisy_row)
    
    augmented_angle_df = pd.DataFrame(augmented_angle_data, columns=angle_df.columns)
    
    # 保存
    seq_output_path = os.path.join(output_dir, 'action_dataset_boxingvi_seq_augmented.csv')
    angle_output_path = os.path.join(output_dir, 'action_dataset_boxingvi_angle_augmented.csv')
    
    augmented_seq_df.to_csv(seq_output_path, index=False)
    augmented_angle_df.to_csv(angle_output_path, index=False)
    
    print(f"\n✅ 数据增强完成！")
    print(f"   原始时序样本数: {len(seq_df)}")
    print(f"   增强时序样本数: {len(augmented_seq_df)}")
    print(f"   原始角度样本数: {len(angle_df)}")
    print(f"   增强角度样本数: {len(augmented_angle_df)}")
    print(f"   增强比例: {len(augmented_seq_df)/len(seq_df):.1f}x")
    print(f"\n📁 保存位置:")
    print(f"   时序特征: {seq_output_path}")
    print(f"   角度特征: {angle_output_path}")
    
    return augmented_seq_df, augmented_angle_df


if __name__ == "__main__":
    seq_data_path = r"D:\boxing_ai_project\model\action_dataset_boxingvi_seq.csv"
    angle_data_path = r"D:\boxing_ai_project\model\action_dataset_boxingvi_angle.csv"
    output_dir = r"D:\boxing_ai_project\model"
    
    if os.path.exists(seq_data_path) and os.path.exists(angle_data_path):
        augment_boxing_dataset(seq_data_path, angle_data_path, output_dir, augment_factor=3)
    else:
        print("❌ 找不到原始数据集文件！请先运行数据集转换程序。")