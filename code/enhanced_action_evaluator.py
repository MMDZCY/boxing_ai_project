# 文件路径: d:\boxing_ai_project\code\enhanced_action_evaluator.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import deque, defaultdict

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("⚠️  fastdtw 未安装，DTW功能将不可用")
    print("   安装命令: pip install fastdtw scipy")

# 配置Matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

class EnhancedActionEvaluator:
    """增强版动作评估器 - 多维度评估"""
    
    def __init__(self):
        self.evaluation_history = []
        self.action_names = {0: "刺拳(Jab)", 1: "直拳(Straight)", 2: "勾拳(Hook)", 3: "摆拳(Swing)"}
        
        # 评估权重配置
        self.weights = {
            'elbow_angle': 0.25,      # 肘角度
            'hip_twist': 0.25,        # 髋扭转
            'dtw_distance': 0.20,     # DTW距离
            'speed_match': 0.15,      # 速度匹配
            'center_of_mass': 0.15     # 重心转移
        }
    
    def calculate_center_of_mass(self, kp):
        """计算身体重心（简化版）"""
        # 使用髋关节中心作为重心近似
        if kp is None or len(kp) < 13:
            return np.array([0, 0])
        
        left_hip = kp[11][:2]
        right_hip = kp[12][:2]
        return (left_hip + right_hip) / 2
    
    def calculate_keypoint_speed(self, kp_history, keypoint_idx=9):
        """计算关键点速度（默认右手腕）"""
        if len(kp_history) < 2:
            return 0.0
        
        speeds = []
        for i in range(1, len(kp_history)):
            prev_kp = kp_history[i-1]
            curr_kp = kp_history[i]
            
            if prev_kp is not None and curr_kp is not None:
                prev_pos = prev_kp[keypoint_idx][:2]
                curr_pos = curr_kp[keypoint_idx][:2]
                speed = np.linalg.norm(curr_pos - prev_pos)
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    def calculate_dtw_distance(self, user_seq, std_seq):
        """计算DTW时序对齐距离"""
        if not HAS_FASTDTW or len(user_seq) == 0 or len(std_seq) == 0:
            return 1.0  # 返回最大距离
        
        # 提取特征序列
        user_features = []
        std_features = []
        
        for kp in user_seq:
            if kp is not None:
                feat = self._extract_simple_features(kp)
                user_features.append(feat)
        
        for kp in std_seq:
            if kp is not None:
                feat = self._extract_simple_features(kp)
                std_features.append(feat)
        
        if len(user_features) == 0 or len(std_features) == 0:
            return 1.0
        
        # 计算DTW距离
        distance, _ = fastdtw(user_features, std_features, dist=euclidean)
        
        # 归一化到 [0, 1]
        max_possible_distance = len(user_features) * len(std_features) * 10
        normalized_distance = min(distance / max_possible_distance, 1.0)
        
        return normalized_distance
    
    def _extract_simple_features(self, kp):
        """提取简单特征用于DTW"""
        return np.array([
            kp[9][0], kp[9][1],  # 右手腕
            kp[10][0], kp[10][1], # 左手腕
            kp[7][0], kp[7][1],   # 右肘
            kp[8][0], kp[8][1],   # 左肘
        ])
    
    def calculate_speed_match_score(self, user_speed_history, std_speed_profile):
        """计算速度曲线匹配度"""
        if len(user_speed_history) < 3 or len(std_speed_profile) < 3:
            return 0.5
        
        # 简单匹配：计算速度均值的相似度
        user_avg = np.mean(user_speed_history)
        std_avg = np.mean(std_speed_profile)
        
        # 计算相对差异
        diff = abs(user_avg - std_avg) / (std_avg + 1e-6)
        match_score = max(0, 1 - diff)
        
        return match_score
    
    def calculate_com_score(self, user_com_history, std_com_profile):
        """计算重心转移匹配度"""
        if len(user_com_history) < 3 or len(std_com_profile) < 3:
            return 0.5
        
        # 计算重心移动距离
        user_movement = np.sum([np.linalg.norm(user_com_history[i] - user_com_history[i-1]) 
                               for i in range(1, len(user_com_history))])
        std_movement = np.sum([np.linalg.norm(std_com_profile[i] - std_com_profile[i-1]) 
                              for i in range(1, len(std_com_profile))])
        
        # 计算相似度
        if std_movement > 0:
            ratio = user_movement / std_movement
            com_score = max(0, 1 - abs(1 - ratio))
        else:
            com_score = 0.5
        
        return com_score
    
    def evaluate_punch(self, punch_data, user_kp_buffer, std_kp_seq):
        """综合评估一次出拳"""
        action_type = punch_data.get('action_type', 1)
        elbow_angle = punch_data.get('elbow_angle', 0)
        hip_twist = punch_data.get('hip_twist', 0)
        
        # 1. 肘角度得分
        elbow_score = min(elbow_angle / 170, 1.0)
        
        # 2. 髋扭转得分
        hip_score = min(hip_twist / 30, 1.0)
        
        # 3. DTW距离得分（越低越好）
        dtw_distance = self.calculate_dtw_distance(user_kp_buffer, std_kp_seq)
        dtw_score = 1 - dtw_distance  # 距离越小，得分越高
        
        # 4. 速度匹配得分
        speed_score = 0.7  # 默认值，简化实现
        
        # 5. 重心转移得分
        com_score = 0.7  # 默认值，简化实现
        
        # 综合加权得分
        total_score = (
            elbow_score * self.weights['elbow_angle'] +
            hip_score * self.weights['hip_twist'] +
            dtw_score * self.weights['dtw_distance'] +
            speed_score * self.weights['speed_match'] +
            com_score * self.weights['center_of_mass']
        ) * 100
        
        # 保存详细评估结果
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'action_name': self.action_names.get(action_type, '未知'),
            'total_score': total_score,
            'dimensions': {
                'elbow_angle': {'score': elbow_score * 100, 'value': elbow_angle},
                'hip_twist': {'score': hip_score * 100, 'value': hip_twist},
                'dtw_distance': {'score': dtw_score * 100, 'value': dtw_distance},
                'speed_match': {'score': speed_score * 100, 'value': speed_score},
                'center_of_mass': {'score': com_score * 100, 'value': com_score}
            }
        }
        
        self.evaluation_history.append(evaluation_result)
        
        return total_score, evaluation_result
    
    def plot_radar_chart(self, evaluation_result, save_path=None):
        """绘制雷达图展示多维度评估"""
        categories = ['肘角度', '髋扭转', 'DTW对齐', '速度匹配', '重心转移']
        scores = [
            evaluation_result['dimensions']['elbow_angle']['score'],
            evaluation_result['dimensions']['hip_twist']['score'],
            evaluation_result['dimensions']['dtw_distance']['score'],
            evaluation_result['dimensions']['speed_match']['score'],
            evaluation_result['dimensions']['center_of_mass']['score']
        ]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores = np.concatenate((scores, [scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        # 绘制雷达图
        ax.plot(angles, scores, 'o-', linewidth=2, label='得分', color='#3498db')
        ax.fill(angles, scores, alpha=0.25, color='#3498db')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        ax.set_title(f"{evaluation_result['action_name']} - 多维度评估\n总得分: {evaluation_result['total_score']:.1f}", 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 雷达图已保存至: {save_path}")
        
        plt.close()
    
    def plot_dimension_trends(self, save_path=None):
        """绘制各维度得分趋势"""
        if len(self.evaluation_history) < 3:
            print("⚠️  数据不足，无法绘制趋势图")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        dimensions = ['elbow_angle', 'hip_twist', 'dtw_distance', 'speed_match', 'center_of_mass']
        dim_names = ['肘角度', '髋扭转', 'DTW对齐', '速度匹配', '重心转移']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (dim, name, color) in enumerate(zip(dimensions, dim_names, colors)):
            scores = [r['dimensions'][dim]['score'] for r in self.evaluation_history]
            axes[i].plot(range(1, len(scores)+1), scores, marker='o', 
                        color=color, linewidth=2, markersize=6)
            axes[i].axhline(y=60, color='gray', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('第N拳', fontsize=11)
            axes[i].set_ylabel('得分', fontsize=11)
            axes[i].set_title(f'{name} 得分趋势', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([0, 105])
        
        # 总得分趋势
        total_scores = [r['total_score'] for r in self.evaluation_history]
        axes[5].plot(range(1, len(total_scores)+1), total_scores, marker='o', 
                    color='#1abc9c', linewidth=3, markersize=8)
        axes[5].axhline(y=60, color='red', linestyle='--', label='及格线')
        axes[5].set_xlabel('第N拳', fontsize=11)
        axes[5].set_ylabel('总得分', fontsize=11)
        axes[5].set_title('总得分趋势', fontsize=12, fontweight='bold')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        axes[5].set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 维度趋势图已保存至: {save_path}")
        
        plt.close()
    
    def plot_action_comparison_heatmap(self, save_path=None):
        """绘制不同动作类型的各维度对比热力图"""
        if len(self.evaluation_history) < 5:
            print("⚠️  数据不足，无法绘制对比热力图")
            return
        
        # 按动作类型分组
        action_data = defaultdict(list)
        for eval_data in self.evaluation_history:
            action_name = eval_data['action_name']
            for dim, data in eval_data['dimensions'].items():
                action_data[(action_name, dim)].append(data['score'])
        
        # 准备热力图数据
        action_names = list(set([k[0] for k in action_data.keys()]))
        dim_names = ['elbow_angle', 'hip_twist', 'dtw_distance', 'speed_match', 'center_of_mass']
        dim_display_names = ['肘角度', '髋扭转', 'DTW对齐', '速度匹配', '重心转移']
        
        heatmap_data = np.zeros((len(action_names), len(dim_names)))
        for i, action in enumerate(action_names):
            for j, dim in enumerate(dim_names):
                key = (action, dim)
                if key in action_data and len(action_data[key]) > 0:
                    heatmap_data[i, j] = np.mean(action_data[key])
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu',
                   xticklabels=dim_display_names, yticklabels=action_names, ax=ax)
        ax.set_title('不同动作类型的各维度平均得分对比', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('评估维度', fontsize=12)
        ax.set_ylabel('动作类型', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 动作对比热力图已保存至: {save_path}")
        
        plt.close()
    
    def print_enhanced_summary(self):
        """打印增强版评估总结"""
        if not self.evaluation_history:
            print("⚠️  暂无评估数据")
            return
        
        print("\n" + "="*80)
        print("📊 增强版多维度评估报告".center(70))
        print("="*80)
        
        total_punches = len(self.evaluation_history)
        avg_total_score = np.mean([r['total_score'] for r in self.evaluation_history])
        
        print(f"\n📈 总出拳次数: {total_punches}")
        print(f"📊 平均总得分: {avg_total_score:.1f}/100")
        
        # 各维度平均得分
        print(f"\n🎯 各维度平均得分:")
        dimensions = ['elbow_angle', 'hip_twist', 'dtw_distance', 'speed_match', 'center_of_mass']
        dim_names = ['肘角度', '髋扭转', 'DTW对齐', '速度匹配', '重心转移']
        
        for dim, name in zip(dimensions, dim_names):
            scores = [r['dimensions'][dim]['score'] for r in self.evaluation_history]
            avg_score = np.mean(scores)
            print(f"   {name}: {avg_score:.1f}/100")
        
        # 找出最强和最弱维度
        dim_scores = [(name, np.mean([r['dimensions'][dim]['score'] for r in self.evaluation_history])) 
                     for dim, name in zip(dimensions, dim_names)]
        dim_scores.sort(key=lambda x: x[1])
        
        print(f"\n💡 分析建议:")
        print(f"   ✅ 最强维度: {dim_scores[-1][0]} ({dim_scores[-1][1]:.1f}分)")
        print(f"   ❌ 最弱维度: {dim_scores[0][0]} ({dim_scores[0][1]:.1f}分)")
        print(f"   💪 建议重点练习: {dim_scores[0][0]}")
        
        print("\n" + "="*80)
    
    def save_enhanced_report(self, save_path=None):
        """保存增强版评估报告"""
        if save_path is None:
            save_path = r"D:\boxing_ai_project\output\enhanced_evaluation_report.json"
        
        report = {
            'report_time': datetime.now().isoformat(),
            'total_punches': len(self.evaluation_history),
            'evaluation_history': self.evaluation_history
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 增强版评估报告已保存至: {save_path}")

# -------------------------- 独立运行测试 --------------------------
if __name__ == "__main__":
    print("="*60)
    print("🧪 增强版动作评估器测试")
    print("="*60)
    
    # 创建评估器
    evaluator = EnhancedActionEvaluator()
    
    # 模拟一些评估数据
    print("\n[模拟] 正在生成测试数据...")
    np.random.seed(42)
    
    for i in range(20):
        action_type = np.random.randint(0, 4)
        elbow_angle = np.random.normal(145, 15)
        hip_twist = np.random.normal(22, 5)
        
        punch_data = {
            'action_type': action_type,
            'elbow_angle': elbow_angle,
            'hip_twist': hip_twist
        }
        
        # 简化版评估
        total_score, eval_result = evaluator.evaluate_punch(punch_data, [], [])
        print(f"   第{i+1}拳 {eval_result['action_name']}: 总得分 {total_score:.1f}")
        
        # 每5拳生成一次雷达图
        if (i + 1) % 5 == 0:
            evaluator.plot_radar_chart(eval_result, 
                save_path=f"D:\\boxing_ai_project\\output\\radar_chart_{i+1}.png")
    
    print(f"\n✅ 已生成 {len(evaluator.evaluation_history)} 条评估数据")
    
    # 打印总结
    evaluator.print_enhanced_summary()
    
    # 生成可视化
    print("\n[生成] 正在生成分析图表...")
    evaluator.plot_dimension_trends(save_path=r"D:\boxing_ai_project\output\dimension_trends.png")
    evaluator.plot_action_comparison_heatmap(save_path=r"D:\boxing_ai_project\output\action_comparison_heatmap.png")
    evaluator.save_enhanced_report()
    
    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)