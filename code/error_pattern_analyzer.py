# 文件路径: d:\boxing_ai_project\code\error_pattern_analyzer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from datetime import datetime
import json

# 配置Matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

class ErrorPatternAnalyzer:
    """错误模式识别与分析器"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.punch_history = []
        
        # 错误阈值定义
        self.error_thresholds = {
            'elbow_angle': {'threshold': 150, 'name': '出拳未伸直', 'weight': 0.4},
            'hip_twist': {'threshold': 20, 'name': '转髋幅度不足', 'weight': 0.4},
            'wrist_height': {'threshold': -0.3, 'name': '手腕位置过低', 'weight': 0.2}
        }
        
        self.action_names = {0: "刺拳(Jab)", 1: "直拳(Straight)", 2: "勾拳(Hook)", 3: "摆拳(Swing)"}
    
    def record_punch(self, punch_data):
        """记录一次出拳数据"""
        punch_data['timestamp'] = datetime.now().isoformat()
        self.punch_history.append(punch_data)
        
        # 限制历史记录数量
        if len(self.punch_history) > self.max_history:
            self.punch_history.pop(0)
    
    def analyze_errors(self, action_filter=None):
        """分析错误模式
        
        Args:
            action_filter: 动作类型过滤，None表示所有动作
        """
        if not self.punch_history:
            return None
        
        # 过滤数据
        filtered_data = self.punch_history
        if action_filter is not None:
            filtered_data = [p for p in self.punch_history if p.get('action_type') == action_filter]
        
        if not filtered_data:
            return None
        
        # 统计错误
        error_stats = defaultdict(lambda: {'count': 0, 'total': 0, 'avg_value': 0})
        
        for punch in filtered_data:
            for error_key, config in self.error_thresholds.items():
                if error_key in punch:
                    error_stats[error_key]['total'] += 1
                    value = punch[error_key]
                    error_stats[error_key]['avg_value'] += value
                    
                    # 判断是否为错误
                    if error_key == 'elbow_angle' and value < config['threshold']:
                        error_stats[error_key]['count'] += 1
                    elif error_key == 'hip_twist' and value < config['threshold']:
                        error_stats[error_key]['count'] += 1
                    elif error_key == 'wrist_height' and value < config['threshold']:
                        error_stats[error_key]['count'] += 1
        
        # 计算百分比
        result = {}
        for error_key, stats in error_stats.items():
            if stats['total'] > 0:
                result[error_key] = {
                    'name': self.error_thresholds[error_key]['name'],
                    'error_rate': stats['count'] / stats['total'],
                    'error_count': stats['count'],
                    'total_count': stats['total'],
                    'avg_value': stats['avg_value'] / stats['total'],
                    'threshold': self.error_thresholds[error_key]['threshold'],
                    'weight': self.error_thresholds[error_key]['weight']
                }
        
        # 按错误率排序
        sorted_errors = sorted(result.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        
        return {
            'total_punches': len(filtered_data),
            'errors': dict(sorted_errors),
            'top_error': sorted_errors[0] if sorted_errors else None
        }
    
    def get_action_specific_analysis(self):
        """按动作类型分别分析"""
        results = {}
        for action_type in range(4):
            analysis = self.analyze_errors(action_filter=action_type)
            if analysis:
                results[self.action_names[action_type]] = analysis
        return results
    
    def generate_improvement_suggestions(self):
        """生成改进建议"""
        analysis = self.analyze_errors()
        if not analysis:
            return ["暂无足够数据进行分析"]
        
        suggestions = []
        top_error = analysis['top_error']
        
        if top_error:
            error_key, error_info = top_error
            
            # 根据主要错误生成建议
            if error_key == 'elbow_angle':
                suggestions.append("🎯 主要问题：出拳手臂未完全伸直")
                suggestions.append("   💡 建议：想象前方有一个目标，全力伸直手臂去击打")
                suggestions.append("   📝 练习：对着镜子练习直拳，注意观察手臂是否完全伸直")
            elif error_key == 'hip_twist':
                suggestions.append("🎯 主要问题：转髋幅度不足")
                suggestions.append("   💡 建议：出拳时发力从脚开始，通过转髋传递力量")
                suggestions.append("   📝 练习：先练习空转髋，感受髋部扭转的发力感")
            elif error_key == 'wrist_height':
                suggestions.append("🎯 主要问题：手腕位置过低")
                suggestions.append("   💡 建议：保持手腕与肩同高或略高于肩")
                suggestions.append("   📝 练习：想象手上托着一杯水，出拳时不要洒出来")
        
        # 总体建议
        if analysis['total_punches'] >= 10:
            avg_score = np.mean([p['score'] for p in self.punch_history[-10:]])
            if avg_score < 50:
                suggestions.append("\n📊 总体建议：先从基础动作开始，不求快但求标准")
            elif avg_score < 70:
                suggestions.append("\n📊 总体建议：继续巩固基础，逐步提高速度和力量")
            else:
                suggestions.append("\n📊 总体建议：动作已经很标准了，可以尝试组合拳练习")
        
        return suggestions
    
    def plot_error_analysis(self, save_path="../output/error_analysis.png"):
        """绘制错误分析图"""
        analysis = self.analyze_errors()
        if not analysis:
            print("⚠️  暂无数据可分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 错误率柱状图
        error_names = []
        error_rates = []
        for error_key, error_info in analysis['errors'].items():
            error_names.append(error_info['name'])
            error_rates.append(error_info['error_rate'] * 100)
        
        axes[0, 0].barh(error_names, error_rates, color='#e74c3c', alpha=0.7)
        axes[0, 0].set_xlabel('错误率 (%)', fontsize=12)
        axes[0, 0].set_title('各类错误发生率', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        axes[0, 0].set_xlim([0, 100])
        
        # 2. 得分趋势图
        if len(self.punch_history) > 0:
            scores = [p['score'] for p in self.punch_history]
            axes[0, 1].plot(range(1, len(scores)+1), scores, marker='o', 
                           color='#3498db', linewidth=2, markersize=6)
            axes[0, 1].axhline(y=60, color='#27ae60', linestyle='--', label='及格线')
            axes[0, 1].set_xlabel('第N拳', fontsize=12)
            axes[0, 1].set_ylabel('得分', fontsize=12)
            axes[0, 1].set_title('得分趋势', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 85])
        
        # 3. 肘角度分布图
        elbow_angles = [p['elbow_angle'] for p in self.punch_history if 'elbow_angle' in p]
        if elbow_angles:
            axes[1, 0].hist(elbow_angles, bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
            axes[1, 0].axvline(x=150, color='#e74c3c', linestyle='--', linewidth=2, label='标准线 (150°)')
            axes[1, 0].set_xlabel('肘角度 (°)', fontsize=12)
            axes[1, 0].set_ylabel('频次', fontsize=12)
            axes[1, 0].set_title('肘关节角度分布', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 转髋角度分布图
        hip_twists = [p['hip_twist'] for p in self.punch_history if 'hip_twist' in p]
        if hip_twists:
            axes[1, 1].hist(hip_twists, bins=20, alpha=0.7, color='#1abc9c', edgecolor='black')
            axes[1, 1].axvline(x=20, color='#e74c3c', linestyle='--', linewidth=2, label='标准线 (20°)')
            axes[1, 1].set_xlabel('转髋角度 (°)', fontsize=12)
            axes[1, 1].set_ylabel('频次', fontsize=12)
            axes[1, 1].set_title('肩髋扭转角度分布', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 错误分析图已保存至: {save_path}")
        plt.close()
    
    def plot_action_comparison(self, save_path="../output/action_comparison.png"):
        """绘制不同动作类型的对比分析"""
        action_analysis = self.get_action_specific_analysis()
        if not action_analysis:
            print("⚠️  暂无足够数据进行动作对比分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 准备数据
        action_names = list(action_analysis.keys())
        avg_scores = [np.mean([p['score'] for p in self.punch_history 
                               if p.get('action_type') == i]) for i in range(4) 
                      if i in [list(self.action_names.keys())[list(self.action_names.values()).index(name)] 
                               for name in action_names]]
        
        # 1. 各动作平均得分
        axes[0, 0].bar(action_names, avg_scores, color='#3498db', alpha=0.7)
        axes[0, 0].set_ylabel('平均得分', fontsize=12)
        axes[0, 0].set_title('各动作类型平均得分', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=15)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 85])
        
        # 2. 各动作错误率热力图
        error_types = list(self.error_thresholds.keys())
        error_matrix = np.zeros((len(action_names), len(error_types)))
        
        for i, action_name in enumerate(action_names):
            analysis = action_analysis[action_name]
            for j, error_type in enumerate(error_types):
                if error_type in analysis['errors']:
                    error_matrix[i, j] = analysis['errors'][error_type]['error_rate'] * 100
        
        sns.heatmap(error_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=[self.error_thresholds[e]['name'] for e in error_types],
                    yticklabels=action_names, ax=axes[0, 1])
        axes[0, 1].set_title('各动作类型错误率 (%)', fontsize=14, fontweight='bold')
        
        # 3. 肘角度按动作分布
        for action_type, action_name in self.action_names.items():
            elbow_data = [p['elbow_angle'] for p in self.punch_history 
                         if p.get('action_type') == action_type and 'elbow_angle' in p]
            if elbow_data:
                axes[1, 0].hist(elbow_data, alpha=0.5, bins=15, label=action_name, density=True)
        axes[1, 0].axvline(x=150, color='red', linestyle='--', label='标准线')
        axes[1, 0].set_xlabel('肘角度 (°)', fontsize=12)
        axes[1, 0].set_ylabel('密度', fontsize=12)
        axes[1, 0].set_title('各动作肘角度分布', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 转髋角度按动作分布
        for action_type, action_name in self.action_names.items():
            hip_data = [p['hip_twist'] for p in self.punch_history 
                       if p.get('action_type') == action_type and 'hip_twist' in p]
            if hip_data:
                axes[1, 1].hist(hip_data, alpha=0.5, bins=15, label=action_name, density=True)
        axes[1, 1].axvline(x=20, color='red', linestyle='--', label='标准线')
        axes[1, 1].set_xlabel('转髋角度 (°)', fontsize=12)
        axes[1, 1].set_ylabel('密度', fontsize=12)
        axes[1, 1].set_title('各动作转髋角度分布', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 动作对比分析图已保存至: {save_path}")
        plt.close()
    
    def save_report(self, save_path="../output/error_analysis_report.json"):
        """保存分析报告为JSON"""
        report = {
            'analysis_time': datetime.now().isoformat(),
            'total_punches': len(self.punch_history),
            'overall_analysis': self.analyze_errors(),
            'action_specific_analysis': self.get_action_specific_analysis(),
            'suggestions': self.generate_improvement_suggestions()
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ 分析报告已保存至: {save_path}")
    
    def print_summary(self):
        """打印总结报告"""
        print("\n" + "="*80)
        print("📊 错误模式识别分析报告".center(70))
        print("="*80)
        
        analysis = self.analyze_errors()
        if not analysis:
            print("⚠️  暂无训练数据，请先进行训练！")
            return
        
        print(f"\n📈 总出拳次数: {analysis['total_punches']}")
        
        if analysis['total_punches'] > 0:
            avg_score = np.mean([p['score'] for p in self.punch_history])
            print(f"📊 平均得分: {avg_score:.1f}/80")
        
        print(f"\n🔍 错误统计:")
        for error_key, error_info in analysis['errors'].items():
            print(f"   ❌ {error_info['name']}: {error_info['error_rate']*100:.1f}% "
                  f"({error_info['error_count']}/{error_info['total_count']})")
            print(f"      平均值: {error_info['avg_value']:.1f} "
                  f"(标准: ≥{error_info['threshold']})")
        
        print(f"\n💡 改进建议:")
        suggestions = self.generate_improvement_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {suggestion}")
        
        print("\n" + "="*80)

# -------------------------- 独立运行测试 --------------------------
if __name__ == "__main__":
    print("="*60)
    print("🧪 错误模式识别模块测试")
    print("="*60)
    
    # 创建分析器
    analyzer = ErrorPatternAnalyzer()
    
    # 模拟一些训练数据
    print("\n[模拟] 正在生成测试数据...")
    np.random.seed(42)
    
    for i in range(50):
        action_type = np.random.randint(0, 4)
        elbow_angle = np.random.normal(140, 15)  # 略低于标准
        hip_twist = np.random.normal(18, 5)       # 略低于标准
        wrist_height = np.random.normal(-0.2, 0.15)
        
        # 计算得分
        elbow_score = min(elbow_angle / 170 * 40, 40)
        twist_score = min(hip_twist / 30 * 40, 40)
        total_score = elbow_score + twist_score
        
        analyzer.record_punch({
            'action_type': action_type,
            'elbow_angle': elbow_angle,
            'hip_twist': hip_twist,
            'wrist_height': wrist_height,
            'score': total_score
        })
    
    print(f"✅ 已生成 {len(analyzer.punch_history)} 条测试数据")
    
    # 打印分析报告
    analyzer.print_summary()
    
    # 生成可视化
    print("\n[生成] 正在生成分析图表...")
    analyzer.plot_error_analysis()
    analyzer.plot_action_comparison()
    
    # 保存报告
    analyzer.save_report()
    
    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)