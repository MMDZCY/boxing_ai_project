import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# -------------------------- 全局配置 --------------------------
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MODEL = YOLO('yolov8n-pose.pt')
CONF_THRESHOLD = 0.5

WINDOW_SIZE = 15
PUNCH_SPEED_THRESHOLD = 12
COOLDOWN_FRAMES = 30
MAX_HISTORY_LENGTH = 10 # 最多记录10拳

# YOLOv8 定义
KEYPOINTS = {
    "nose":0, "left_shoulder":5, "right_shoulder":6,
    "left_elbow":7, "right_elbow":8, "left_wrist":9, "right_wrist":10,
    "left_hip":11, "right_hip":12
}
COLOR_STANDARD = (46, 204, 113)
COLOR_USER = (231, 76, 60)
COLOR_WARN = (241, 196, 15)
COLOR_WHITE = (255,255,255)
COLOR_PUNCH = (155, 89, 182)

# 配置Matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def put_chinese_text(img, text, position, font_size=24, color=COLOR_WHITE):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def calculate_punch_score(elbow_angle, hip_twist_angle):
    elbow_score = min(elbow_angle / 170 * 40, 40)
    twist_score = min(hip_twist_angle / 30 * 40, 40)
    total_score = elbow_score + twist_score
    return total_score, elbow_score, twist_score

def generate_history_plot(score_history):
    """
    生成历史得分趋势图，返回OpenCV格式的图片
    """
    # 创建一个小图
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    
    if not score_history:
        ax.text(0.5, 0.5, "暂无记录，快出拳吧！", ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_ylim(0, 100)
    else:
        x = list(range(1, len(score_history)+1))
        y = score_history
        
        # 绘制折线图
        ax.plot(x, y, marker='o', color='#ee5a24', linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color='#ee5a24')
        
        # 绘制及格线
        ax.axhline(y=60, color='#27ae60', linestyle='--', label='及格线 (60分)')
        
        # 设置坐标轴
        ax.set_title("最近10拳得分趋势", fontsize=12)
        ax.set_xlabel("第N拳", fontsize=10)
        ax.set_ylabel("得分", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(0.5, MAX_HISTORY_LENGTH + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # 将Matplotlib图转换为OpenCV图片
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_plot = np.asarray(buf)
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    plt.close(fig) # 关闭图，防止内存泄漏
    
    return img_plot

# -------------------------- 主程序 --------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("="*50)
    print("🥊  实时拳击动作AI反馈系统 (历史记录版)")
    print("="*50)
    print("📌 新增功能：")
    print("   右上角实时显示【最近10拳得分趋势图】")
    print("="*50)

    # 状态变量
    wrist_history = deque(maxlen=WINDOW_SIZE)
    cooldown_counter = 0
    last_score = None
    last_suggestion = ""
    punch_highlight_frame = 0
    
    # 新增：历史记录变量
    score_history = [] # 存储得分历史
    history_plot_img = generate_history_plot(score_history) # 初始趋势图

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. YOLO检测
        results = MODEL(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()
        keypoints = results[0].keypoints.data

        current_elbow = 0
        current_twist = 0
        punch_detected = False

        if keypoints.shape[0] > 0:
            kp = keypoints[0].cpu().numpy()
            current_wrist_pos = kp[KEYPOINTS["right_wrist"]]
            
            # 计算角度
            shoulder = kp[KEYPOINTS["right_shoulder"]]
            elbow = kp[KEYPOINTS["right_elbow"]]
            wrist = kp[KEYPOINTS["right_wrist"]]
            current_elbow = calculate_angle(shoulder, elbow, wrist)

            left_shoulder = kp[KEYPOINTS["left_shoulder"]]
            right_shoulder = kp[KEYPOINTS["right_shoulder"]]
            left_hip = kp[KEYPOINTS["left_hip"]]
            right_hip = kp[KEYPOINTS["right_hip"]]
            shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
            hip_vec = right_hip[:2] - left_hip[:2]
            current_twist = calculate_angle(np.array([0,0]), shoulder_vec, hip_vec)

            # 2. 自动出拳检测
            wrist_history.append(current_wrist_pos)
            
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                if len(wrist_history) == WINDOW_SIZE:
                    start_pos = np.mean([p[:2] for p in list(wrist_history)[:5]], axis=0)
                    end_pos = np.mean([p[:2] for p in list(wrist_history)[-5:]], axis=0)
                    movement_dist = np.linalg.norm(end_pos - start_pos)
                    
                    if movement_dist > PUNCH_SPEED_THRESHOLD and current_elbow > 100:
                        punch_detected = True
                        cooldown_counter = COOLDOWN_FRAMES
                        punch_highlight_frame = 15
                        
                        # 计算得分
                        total, elbow_s, twist_s = calculate_punch_score(current_elbow, current_twist)
                        last_score = total
                        
                        # 新增：更新历史记录
                        score_history.append(total)
                        if len(score_history) > MAX_HISTORY_LENGTH:
                            score_history.pop(0) # 超过10个就把最老的删掉
                        # 重新生成趋势图
                        history_plot_img = generate_history_plot(score_history)
                        
                        # 生成建议
                        suggs = []
                        if current_elbow < 150:
                            suggs.append("❌ 出拳未完全伸直")
                        if current_twist < 20:
                            suggs.append("❌ 转髋幅度不足")
                        if not suggs:
                            suggs.append("✅ 动作标准！")
                        last_suggestion = " | ".join(suggs)
                        
                        print(f"🎯 第{len(score_history)}拳 | 得分：{total:.1f}/80 | {last_suggestion}")

       
        # --- 左上角：实时指标 ---
        cv2.rectangle(annotated_frame, (10, 10), (420, 200), (0,0,0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (420, 200), COLOR_WHITE, 2)
        
        status_text = "状态：等待出拳..." if cooldown_counter == 0 else "状态：冷却中..."
        status_color = COLOR_WHITE if cooldown_counter == 0 else COLOR_WARN
        
        annotated_frame = put_chinese_text(annotated_frame, "🥊 实时动作指标", (20, 15), font_size=20)
        annotated_frame = put_chinese_text(annotated_frame, f"右肘关节角度：{current_elbow:.1f}°", (20, 50), font_size=18)
        annotated_frame = put_chinese_text(annotated_frame, f"肩髋扭转角度：{current_twist:.1f}°", (20, 85), font_size=18)
        annotated_frame = put_chinese_text(annotated_frame, status_text, (20, 120), font_size=16, color=status_color)

    
        # 把生成的Matplotlib图贴到画面上
        h_plot, w_plot = history_plot_img.shape[:2]
        # 位置：右上角，留一点边距
        x_offset = FRAME_WIDTH - w_plot - 20
        y_offset = 20
        # 叠加图片
        annotated_frame[y_offset:y_offset+h_plot, x_offset:x_offset+w_plot] = history_plot_img

        # --- 出拳高亮 ---
        if punch_highlight_frame > 0:
            punch_highlight_frame -= 1
            cv2.circle(annotated_frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2), 200, COLOR_PUNCH, 10)
            cv2.putText(annotated_frame, "PUNCH!", (FRAME_WIDTH//2 - 100, FRAME_HEIGHT//2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, COLOR_PUNCH, 5, cv2.LINE_AA)

        # --- 右下角：最近一次打分 ---
        if last_score is not None:
            cv2.rectangle(annotated_frame, (FRAME_WIDTH-420, FRAME_HEIGHT-200), (FRAME_WIDTH-20, FRAME_HEIGHT-20), (0,0,0), -1)
            cv2.rectangle(annotated_frame, (FRAME_WIDTH-420, FRAME_HEIGHT-200), (FRAME_WIDTH-20, FRAME_HEIGHT-20), COLOR_WHITE, 2)
            
            score_color = COLOR_STANDARD if last_score >= 60 else COLOR_WARN
            
            annotated_frame = put_chinese_text(annotated_frame, "📊 最近一次出拳", (FRAME_WIDTH-400, FRAME_HEIGHT-185), font_size=20)
            annotated_frame = put_chinese_text(annotated_frame, f"总得分：{last_score:.1f}/80", (FRAME_WIDTH-400, FRAME_HEIGHT-150), font_size=24, color=score_color)
            annotated_frame = put_chinese_text(annotated_frame, last_suggestion, (FRAME_WIDTH-400, FRAME_HEIGHT-110), font_size=16)

        cv2.imshow("🥊 拳击动作AI实时反馈", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✅ 训练结束！")
            if score_history:
                print(f"📊 本次训练统计：")
                print(f"   总出拳次数：{len(score_history)}")
                print(f"   平均得分：{np.mean(score_history):.1f}")
                print(f"   最高得分：{np.max(score_history):.1f}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()