import numpy as np

skeleton_path = r"D:\boxing_ai_project\boxingvi_dataset\Skeleton_data\V1.npy"
data = np.load(skeleton_path)

print("="*60)
print(f"V1.npy 的形状：{data.shape}")
print(f"数据类型：{data.dtype}")
print("\n第0帧的数据（前10个关键点）：")
print(data[0][:10])
print("\n第0帧的完整形状：", data[0].shape)
print("="*60)