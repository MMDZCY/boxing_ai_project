import pandas as pd
import os

# 改成你的绝对路径
annotation_path = r"D:\boxing_ai_project\boxingvi_dataset\Annotation_files\V1.xlsx"

df = pd.read_excel(annotation_path)
print("="*60)
print("V1.xlsx 的前10行内容：")
print(df.head(10))
print("\n" + "="*60)
print("所有列名：")
print(df.columns.tolist())
print("\n" + "="*60)
print("标签列的所有唯一值：")
# 尝试找标签列
for col in df.columns:
    if 'class' in col.lower() or 'label' in col.lower() or 'punch' in col.lower():
        print(f"\n列名：{col}")
        print(df[col].value_counts())
print("="*60)