# 文件路径: d:\boxing_ai_project\code\train_advanced_models.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Conv1D, GlobalAveragePooling1D, Bidirectional, GRU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 配置 --------------------------
DATASET_PATH = r"D:\boxing_ai_project\model\action_dataset_boxingvi_seq.csv"
MODEL_SAVE_DIR = r"D:\boxing_ai_project\model"
TIME_STEPS = 10
FEATURE_DIM = 34
NUM_CLASSES = 4
ACTION_NAMES = {0: "刺拳", 1: "直拳", 2: "勾拳", 3: "摆拳"}

# -------------------------- 模型定义 --------------------------
class TransformerBlock(tf.keras.layers.Layer):
    """Transformer编码器块"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

def build_lstm_baseline(input_shape, num_classes):
    """原始LSTM模型（与原始脚本一致）"""
    return Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

def build_bi_lstm_model(input_shape, num_classes):
    """双向LSTM模型"""
    return Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(16)),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

def build_gru_model(input_shape, num_classes):
    """GRU模型"""
    return Sequential([
        Input(shape=input_shape),
        GRU(48, return_sequences=False),
        Dropout(0.4),
        Dense(24, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

def build_cnn_lstm_model(input_shape, num_classes):
    """CNN-LSTM混合模型"""
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.4)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def build_transformer_model(input_shape, num_classes):
    """Transformer模型"""
    inputs = Input(shape=input_shape)
    x = Dense(64)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = TransformerBlock(d_model=64, num_heads=4, dff=128, dropout_rate=0.2)(x)
    x = TransformerBlock(d_model=64, num_heads=4, dff=128, dropout_rate=0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# -------------------------- 主程序 --------------------------
def main():
    print("="*70)
    print("🤖 高级模型训练与对比程序")
    print("="*70)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. 加载数据（与原始脚本完全一致）
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ 加载训练集成功：共 {len(df)} 个样本")
    
    # 2. 准备数据（与原始脚本完全一致）
    X = df.iloc[:, :-1].values  # 340维特征
    y = df['label'].values.astype(int)   # 标签
    X, y = shuffle(X, y, random_state=42)
    
    # 3. 重塑为时序数据（与原始脚本完全一致）
    n_samples = len(X)
    X_seq = X.reshape(n_samples, TIME_STEPS, FEATURE_DIM)
    y_seq = y
    
    print(f"✅ 时序数据重塑完成：{X_seq.shape}")
    
    # 4. 划分数据集（与原始脚本完全一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # 5. 标准化（与原始脚本完全一致）
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, FEATURE_DIM)
    X_test_flat = X_test.reshape(-1, FEATURE_DIM)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # 6. 标签独热编码（与原始脚本完全一致）
    y_train_onehot = to_categorical(y_train, num_classes=4)
    y_test_onehot = to_categorical(y_test, num_classes=4)
    
    print(f"\n🔄 数据集划分完成：")
    print(f"   训练集：{len(X_train)} 个时序样本")
    print(f"   测试集：{len(X_test)} 个时序样本")
    
    # 7. 定义所有模型
    input_shape = (TIME_STEPS, FEATURE_DIM)
    models = {
        'LSTM (Baseline)': build_lstm_baseline(input_shape, NUM_CLASSES),
        'Bi-LSTM': build_bi_lstm_model(input_shape, NUM_CLASSES),
        'GRU': build_gru_model(input_shape, NUM_CLASSES),
        'CNN-LSTM': build_cnn_lstm_model(input_shape, NUM_CLASSES),
        'Transformer': build_transformer_model(input_shape, NUM_CLASSES)
    }
    
    # 8. 训练所有模型
    trained_models = {}
    histories = []
    accuracies = []
    model_names = list(models.keys())
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🚀 训练模型: {name}")
        print(f"{'='*60}")
        
        # 编译模型（与原始脚本一致）
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 回调函数（与原始脚本一致）
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
        
        # 训练（与原始脚本一致）
        history = model.fit(
            X_train_scaled, y_train_onehot,
            batch_size=32,
            epochs=100,
            validation_data=(X_test_scaled, y_test_onehot),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 评估（与原始脚本一致）
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ {name} - 测试准确率: {accuracy:.2%}")
        
        # 分类报告
        print(f"\n📊 {name} - 分类报告:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]))
        
        trained_models[name] = model
        histories.append(history)
        accuracies.append(accuracy)
        
        # 保存模型
        model_filename = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        model.save(os.path.join(MODEL_SAVE_DIR, f'{model_filename}.h5'))
    
    # 保存scaler
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler_advanced.pkl'))
    
    # 9. 可视化对比
    print(f"\n{'='*60}")
    print("📊 生成对比图表")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (name, history, color) in enumerate(zip(model_names, histories, colors)):
        axes[0].plot(history.history['accuracy'], label=f'{name} (训练)', 
                    color=color, linestyle='-', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label=f'{name} (验证)', 
                    color=color, linestyle='--', linewidth=2)
        axes[1].plot(history.history['loss'], label=f'{name} (训练)', 
                    color=color, linestyle='-', linewidth=2)
        axes[1].plot(history.history['val_loss'], label=f'{name} (验证)', 
                    color=color, linestyle='--', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('准确率对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('损失函数对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(r"D:\boxing_ai_project\output", exist_ok=True)
    training_plot_path = r"D:\boxing_ai_project\output\model_training_comparison.png"
    plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练对比图已保存至: {training_plot_path}")
    plt.close()
    
    # 柱状图
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('模型类型', fontsize=14)
    ax.set_ylabel('测试准确率', fontsize=14)
    ax.set_title('不同模型的性能对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    accuracy_plot_path = r"D:\boxing_ai_project\output\model_accuracy_comparison.png"
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 模型对比图已保存至: {accuracy_plot_path}")
    plt.close()
    
    # 打印最终对比
    print(f"\n{'='*60}")
    print("🏆 模型性能最终对比")
    print(f"{'='*60}")
    
    result_df = pd.DataFrame({
        '模型': model_names,
        '测试准确率': [f'{acc:.2%}' for acc in accuracies]
    }).sort_values('测试准确率', ascending=False)
    
    print(result_df.to_string(index=False))
    
    # 找出最佳模型
    best_idx = np.argmax(accuracies)
    print(f"\n🏆 最佳模型: {model_names[best_idx]} ({accuracies[best_idx]:.2%})")
    
    # 保存对比结果
    result_df.to_csv(r"D:\boxing_ai_project\output\model_comparison_results.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 所有模型训练完成！")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()