# 诊断脚本：分析为什么高级模型训练准确率低
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("="*80)
print("🔍 诊断脚本：分析高级模型训练准确率问题")
print("="*80)

# 路径配置
DATASET_PATH = r"D:\boxing_ai_project\model\action_dataset_boxingvi_seq.csv"

# =========================== 步骤1：运行原始训练脚本 ===========================
print("\n" + "="*80)
print("📊 步骤1：使用原始训练脚本训练 LSTM 模型")
print("="*80)

def train_original_lstm():
    """原始训练脚本的逻辑"""
    print("加载数据...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ 加载训练集成功：共 {len(df)} 个样本")
    
    # 准备数据
    X = df.iloc[:, :-1].values
    y = df['label'].values.astype(int)
    X, y = shuffle(X, y, random_state=42)
    
    # 重塑为时序数据
    n_samples = len(X)
    X_seq = X.reshape(n_samples, 10, 34)
    y_seq = y
    
    print(f"✅ 时序数据重塑完成：{X_seq.shape}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 34)
    X_test_flat = X_test.reshape(-1, 34)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # 标签独热编码
    y_train_onehot = to_categorical(y_train, num_classes=4)
    y_test_onehot = to_categorical(y_test, num_classes=4)
    
    print(f"\n数据集划分完成：")
    print(f"   训练集：{len(X_train)} 个时序样本")
    print(f"   测试集：{len(X_test)} 个时序样本")
    
    # 构建LSTM模型
    print("\n构建LSTM模型...")
    model = Sequential([
        Input(shape=(10, 34)),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # 训练模型
    print("\n开始训练LSTM模型...")
    history = model.fit(
        X_train_scaled, y_train_onehot,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_scaled, y_test_onehot),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 模型评估
    print("\n模型评估结果：")
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ 原始训练脚本准确率：{accuracy:.2%}")
    
    print("\n" + "="*80)
    return accuracy, X_train_scaled, X_test_scaled, y_train, y_test, scaler

original_acc, X_train_orig, X_test_orig, y_train_orig, y_test_orig, scaler_orig = train_original_lstm()

print("\n\n" + "="*80)
print("🎯 诊断总结：")
print("="*80)
print(f"原始训练脚本准确率：{original_acc:.2%}")
print("\n如果原始准确率在80%+，说明问题在高级脚本的实现上。")
print("如果原始准确率也只有50%，说明数据或模型本身就有问题。")