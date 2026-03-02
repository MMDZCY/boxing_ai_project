# 文件路径: d:\boxing_ai_project\code\train_lstm_classifier.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------- 配置 --------------------------
DATASET_PATH = r"D:\boxing_ai_project\model\action_dataset_boxingvi_seq.csv"
MODEL_SAVE_PATH = r"D:\boxing_ai_project\model\lstm_action_classifier.h5"
SCALER_SAVE_PATH = r"D:\boxing_ai_project\model\lstm_scaler.pkl"
ACTION_NAMES = {0: "jab(刺拳)", 1: "straight(直拳)", 2: "hook(勾拳)", 3: "swing(摆拳)"}
# LSTM参数
TIME_STEPS = 10  # 时序窗口大小
FEATURE_DIM = 34  # 340维 / 10帧 = 34维/帧？需要查看数据实际格式

# -------------------------- 主程序 --------------------------
def main():
    print("="*60)
    print("🥊 拳击动作分类模型训练（LSTM时序版）")
    print("="*60)

    # 1. 加载数据
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ 加载训练集成功：共 {len(df)} 个样本")
    
    # 2. 准备数据
    X = df.iloc[:, :-1].values  # 340维特征
    y = df['label'].values.astype(int)   # 标签
    X, y = shuffle(X, y, random_state=42)
    
    # 3. 重塑为时序数据 (样本数, 时间步, 特征维)
    # 340维 = 10帧 × 17关键点 × 2坐标
    n_samples = len(X)
    X_seq = X.reshape(n_samples, 10, 34)  # (样本, 10帧, 34维)
    y_seq = y
    
    print(f"✅ 时序数据重塑完成：{X_seq.shape}")
    
    # 4. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # 5. 标准化（只在训练集fit）
    scaler = StandardScaler()
    # 展平标准化，再重塑
    X_train_flat = X_train.reshape(-1, 34)
    X_test_flat = X_test.reshape(-1, 34)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # 6. 标签独热编码
    y_train_onehot = to_categorical(y_train, num_classes=4)
    y_test_onehot = to_categorical(y_test, num_classes=4)
    
    print(f"\n🔄 数据集划分完成：")
    print(f"   训练集：{len(X_train)} 个时序样本")
    print(f"   测试集：{len(X_test)} 个时序样本")

    # 7. 构建LSTM模型
    print("\n🚀 构建LSTM模型...")
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
    
    # 8. 设置回调
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
    
    # 9. 训练模型
    print("\n🚀 开始训练LSTM模型...")
    history = model.fit(
        X_train_scaled, y_train_onehot,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_scaled, y_test_onehot),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 10. 模型评估
    print("\n📈 模型评估结果：")
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ 测试集准确率：{accuracy:.2%}")
    
    # 详细报告
    print("\n📋 详细分类报告：")
    print(classification_report(
        y_test, y_pred,
        target_names=[ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]
    ))

    # 11. 保存模型和标准化器
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\n💾 LSTM模型已保存至：{MODEL_SAVE_PATH}")
    print(f"💾 标准化器已保存至：{SCALER_SAVE_PATH}")

    print("\n="*60)
    print("🎉 LSTM模型训练完成！")
    print("="*60)

if __name__ == "__main__":
    # 关闭TensorFlow警告
    tf.get_logger().setLevel('ERROR')
    main()