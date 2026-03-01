import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pyttsx3

# -------------------------- 全局配置 --------------------------
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MODEL = YOLO('yolov8n-pose.pt')
CONF_THRESHOLD = 0.5

# 标准动作路径（和你之前的模板完全兼容）
STANDARD_ACTION_PATH = "../standard_action/standard_straight.npy"

# 检测与匹配配置
WINDOW_SIZE = 15
PUNCH_SPEED_THRESHOLD = 12
COOLDOWN_FRAMES = 30
MAX_HISTORY_LENGTH = 10
# 新增：骨骼匹配平滑窗口（解决跳帧抖动）
MATCH_SMOOTH_WINDOW = 5

# YOLOv8 定义
KEYPOINTS = {
    "nose":0, "left_shoulder":5, "right_shoulder":6,
    "left_elbow":7, "right_elbow":8, "left_wrist":9, "right_wrist":10,
    "left_hip":11, "right_hip":12
}
SKELETON = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]

# 颜色定义 (BGR格式，OpenCV专用)
COLOR_USER = (0, 0, 255)       # 用户骨骼：实色红
COLOR_STD = (0, 255, 0)        # 标准骨骼：半透绿
COLOR_WARN = (0, 230, 255)
COLOR_WHITE = (255,255,255)
COLOR_BLACK = (0,0,0)

# 配置Matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 工具函数 --------------------------
def calculate_angle(p1, p2, p3):
    """和之前完全一致的角度计算，保证逻辑统一"""
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def put_chinese_text(img, text, position, font_size=24, color=COLOR_WHITE):
    """解决中文显示问题"""
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
    """和之前完全一致的打分逻辑，保证结果统一"""
    elbow_score = min(elbow_angle / 170 * 40, 40)
    twist_score = min(hip_twist_angle / 30 * 40, 40)
    total_score = elbow_score + twist_score
    return total_score, elbow_score, twist_score

def generate_history_plot(score_history):
    """历史得分趋势图，和之前完全一致"""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    if not score_history:
        ax.text(0.5, 0.5, "暂无记录，快出拳吧！", ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_ylim(0, 100)
    else:
        x = list(range(1, len(score_history)+1))
        y = score_history
        ax.plot(x, y, marker='o', color='#ee5a24', linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color='#ee5a24')
        ax.axhline(y=60, color='#27ae60', linestyle='--', label='及格线 (60分)')
        ax.set_title("最近10拳得分趋势", fontsize=12)
        ax.set_xlabel("第N拳", fontsize=10)
        ax.set_ylabel("得分", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(0.5, MAX_HISTORY_LENGTH + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_plot = np.asarray(buf)
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_plot

# -------------------------- 新增：骨骼叠加核心优化函数 --------------------------
def extract_action_features(kp):
    """多特征提取，解决单特征匹配不准的问题"""
    # 特征1：右肘关节角度
    elbow_angle = calculate_angle(kp[5], kp[7], kp[9])
    # 特征2：肩髋扭转角度
    shoulder_vec = kp[6][:2] - kp[5][:2]
    hip_vec = kp[12][:2] - kp[11][:2]
    twist_angle = calculate_angle(np.array([0,0]), shoulder_vec, hip_vec)
    # 特征3：右腕相对肩的高度
    wrist_height = (kp[9][1] - kp[5][1]) / (np.linalg.norm(kp[5]-kp[6]) + 1e-6)
    return np.array([elbow_angle, twist_angle, wrist_height])

def preprocess_standard_action(std_kp_seq):
    """预计算标准动作的所有特征，避免实时重复计算，提升流畅度"""
    std_features = []
    for kp in std_kp_seq:
        std_features.append(extract_action_features(kp))
    return np.array(std_features)

def find_smooth_matching_frame(user_feat, std_features, last_idx, smooth_window=5):
    """平滑匹配：解决跳帧抖动问题，限制匹配范围在最近帧的附近"""
    # 限制搜索范围，避免跳帧
    min_search = max(0, last_idx - smooth_window)
    max_search = min(len(std_features)-1, last_idx + smooth_window)
    # 多特征加权匹配（肘关节权重最高）
    weights = np.array([0.6, 0.3, 0.1])
    min_dist = float('inf')
    best_idx = last_idx
    for i in range(min_search, max_search+1):
        dist = np.sum(np.abs(user_feat - std_features[i]) * weights)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

def align_std_to_user_body(std_kp, user_kp):
    """核心优化：把标准骨骼对齐到用户的实际身体位置，而不是画面中心"""
    std_kp_aligned = std_kp.copy()
    
    # 1. 计算用户的身体基准：髋关节中心 + 肩宽
    user_hip_center = (user_kp[11][:2] + user_kp[12][:2]) / 2
    user_shoulder_width = np.linalg.norm(user_kp[5][:2] - user_kp[6][:2])
    
    # 2. 计算标准动作的基准
    std_hip_center = (std_kp[11][:2] + std_kp[12][:2]) / 2
    std_shoulder_width = np.linalg.norm(std_kp[5][:2] - std_kp[6][:2])
    
    # 3. 缩放：和用户的肩宽一致
    if std_shoulder_width > 10:
        scale = user_shoulder_width / std_shoulder_width
        std_kp_aligned[:, :2] *= scale
        std_hip_center *= scale
    
    # 4. 平移：对齐到用户的髋关节中心，完全贴合用户的身体位置
    std_kp_aligned[:, :2] += (user_hip_center - std_hip_center)
    
    return std_kp_aligned

def draw_skeleton_overlay(img, kp, color, alpha=1.0, thickness=3):
    """优化的骨骼绘制，支持半透明，不遮挡用户的身体"""
    overlay = img.copy()
    # 绘制骨骼连线
    for (i, j) in SKELETON:
        pt1 = (int(kp[i][0]), int(kp[i][1]))
        pt2 = (int(kp[j][0]), int(kp[j][1]))
        # 只绘制可见的关键点
        if kp[i][0] > 0 and kp[i][0] < img.shape[1] and kp[j][0] > 0 and kp[j][0] < img.shape[1]:
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    # 绘制关键点
    for i in range(len(kp)):
        pt = (int(kp[i][0]), int(kp[i][1]))
        if kp[i][0] > 0 and kp[i][0] < img.shape[1]:
            cv2.circle(overlay, pt, 6, color, -1, cv2.LINE_AA)
    # 半透明混合
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        img[:] = overlay[:]
    return img

# -------------------------- 主程序 --------------------------
def main():
    # 1. 初始化所有模块
    # 语音引擎初始化
    print("[初始化] 正在启动语音引擎...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)
    
    # 标准动作预加载与预处理
    print("[初始化] 正在加载并预处理标准动作...")
    if not os.path.exists(STANDARD_ACTION_PATH):
        raise FileNotFoundError(f"找不到标准动作文件：{STANDARD_ACTION_PATH}，请先运行 extract_standard_actions.py")
    std_kp_seq = np.load(STANDARD_ACTION_PATH)
    std_features = preprocess_standard_action(std_kp_seq)
    last_match_idx = 0 # 记录上一次匹配的帧，用于平滑
    print(f"✅ 标准动作预处理完成：共 {len(std_kp_seq)} 帧")

    # 摄像头初始化
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 状态变量初始化（和之前完全一致，保证功能兼容）
    wrist_history = deque(maxlen=WINDOW_SIZE)
    cooldown_counter = 0
    last_score = None
    last_suggestion = ""
    last_suggestion_speak = ""
    punch_highlight_frame = 0
    score_history = []
    history_plot_img = generate_history_plot(score_history)
    # 新增：骨骼叠加开关
    skeleton_overlay_enabled = True

    print("="*60)
    print("🥊  拳击动作AI训练系统 (终极完整版)")
    print("="*60)
    print("📌 功能清单：")
    print("   ✅ 实时姿态检测 | ✅ 自动出拳识别 | ✅ 专项动作打分")
    print("   ✅ 语音播报反馈 | ✅ 训练趋势记录 | ✅ 实时骨骼叠加")
    print("📌 操作说明：")
    print("   【Tab键】开启/关闭骨骼叠加 | 【Q键】退出程序")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # 镜像翻转，符合人眼直觉
        display_frame = frame.copy()

        # 2. YOLO实时姿态检测
        results = MODEL(frame, conf=CONF_THRESHOLD, verbose=False)
        keypoints = results[0].keypoints.data
        user_kp = None
        current_elbow = 0
        current_twist = 0

        if keypoints.shape[0] > 0:
            user_kp = keypoints[0].cpu().numpy()
            # 计算核心指标（和之前完全一致）
            current_elbow = calculate_angle(user_kp[5], user_kp[7], user_kp[9])
            left_shoulder, right_shoulder = user_kp[5], user_kp[6]
            left_hip, right_hip = user_kp[11], user_kp[12]
            shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
            hip_vec = right_hip[:2] - left_hip[:2]
            current_twist = calculate_angle(np.array([0,0]), shoulder_vec, hip_vec)

            # -------------------------- 新增：核心骨骼叠加逻辑 --------------------------
            if skeleton_overlay_enabled:
                # 提取用户当前动作特征
                user_feat = extract_action_features(user_kp)
                # 平滑匹配标准动作帧
                best_match_idx = find_smooth_matching_frame(user_feat, std_features, last_match_idx, MATCH_SMOOTH_WINDOW)
                last_match_idx = best_match_idx
                # 取出匹配的标准帧
                matched_std_kp = std_kp_seq[best_match_idx]
                # 把标准骨骼对齐到用户的身体位置
                std_kp_aligned = align_std_to_user_body(matched_std_kp, user_kp)
                # 绘制：先画半透标准绿骨骼，再画用户实色红骨骼
                display_frame = draw_skeleton_overlay(display_frame, std_kp_aligned, COLOR_STD, alpha=0.4, thickness=4)
                display_frame = draw_skeleton_overlay(display_frame, user_kp, COLOR_USER, alpha=1.0, thickness=2)
                # 显示匹配信息
                match_info = f"匹配标准帧：{best_match_idx} | 骨骼叠加：开启"
            else:
                match_info = "骨骼叠加：关闭"
                # 关闭叠加时，只画用户自己的骨骼
                display_frame = draw_skeleton_overlay(display_frame, user_kp, COLOR_USER, alpha=1.0, thickness=2)

            # -------------------------- 原有自动出拳检测逻辑（完全保留） --------------------------
            current_wrist = user_kp[9]
            wrist_history.append(current_wrist)
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                if len(wrist_history) == WINDOW_SIZE:
                    start_pos = np.mean([p[:2] for p in list(wrist_history)[:5]], axis=0)
                    end_pos = np.mean([p[:2] for p in list(wrist_history)[-5:]], axis=0)
                    movement_dist = np.linalg.norm(end_pos - start_pos)
                    # 出拳检测逻辑
                    if movement_dist > PUNCH_SPEED_THRESHOLD and current_elbow > 100:
                        cooldown_counter = COOLDOWN_FRAMES
                        punch_highlight_frame = 15
                        # 计算得分
                        total, _, _ = calculate_punch_score(current_elbow, current_twist)
                        last_score = total
                        # 更新历史记录
                        score_history.append(total)
                        if len(score_history) > MAX_HISTORY_LENGTH:
                            score_history.pop(0)
                        history_plot_img = generate_history_plot(score_history)
                        # 生成建议
                        suggs_text, suggs_speak = [], []
                        if current_elbow < 150:
                            suggs_text.append("❌ 出拳未完全伸直")
                            suggs_speak.append("出拳未完全伸直")
                        else:
                            suggs_text.append("✅ 出拳伸直")
                        if current_twist < 20:
                            suggs_text.append("❌ 转髋幅度不足")
                            suggs_speak.append("转髋幅度不足")
                        else:
                            suggs_text.append("✅ 转髋充分")
                        if not suggs_speak:
                            suggs_speak.append("动作非常标准")
                            suggs_text.append("✅ 动作标准！")
                        last_suggestion = " | ".join(suggs_text)
                        last_suggestion_speak = "。".join(suggs_speak)
                        # 控制台输出
                        print(f"🎯 第{len(score_history)}拳 | 得分：{total:.1f}/80 | {last_suggestion}")
                        # 语音播报
                        speak_content = f"第{len(score_history)}拳，得分{int(total)}分。{last_suggestion_speak}"
                        tts_engine.say(speak_content)
                        tts_engine.runAndWait()

        # 3. UI界面叠加（完全保留原有功能）
        # --- 左上角实时指标面板 ---
        cv2.rectangle(display_frame, (10, 10), (420, 200), COLOR_BLACK, -1)
        cv2.rectangle(display_frame, (10, 10), (420, 200), COLOR_WHITE, 2)
        status_text = "状态：等待出拳..." if cooldown_counter == 0 else "状态：冷却中..."
        status_color = COLOR_WHITE if cooldown_counter == 0 else COLOR_WARN
        display_frame = put_chinese_text(display_frame, "🥊 实时动作指标", (20, 15), font_size=20)
        display_frame = put_chinese_text(display_frame, f"右肘关节角度：{current_elbow:.1f}°", (20, 50), font_size=18)
        display_frame = put_chinese_text(display_frame, f"肩髋扭转角度：{current_twist:.1f}°", (20, 85), font_size=18)
        display_frame = put_chinese_text(display_frame, status_text, (20, 120), font_size=16, color=status_color)
        if 'match_info' in locals():
            display_frame = put_chinese_text(display_frame, match_info, (20, 155), font_size=14, color=COLOR_STD if skeleton_overlay_enabled else COLOR_WHITE)

        # --- 右上角历史趋势图 ---
        h_plot, w_plot = history_plot_img.shape[:2]
        x_offset = FRAME_WIDTH - w_plot - 20
        y_offset = 20
        display_frame[y_offset:y_offset+h_plot, x_offset:x_offset+w_plot] = history_plot_img

        # --- 出拳高亮特效 ---
        if punch_highlight_frame > 0:
            punch_highlight_frame -= 1
            cv2.circle(display_frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2), 200, (155, 89, 182), 10)
            cv2.putText(display_frame, "PUNCH!", (FRAME_WIDTH//2 - 100, FRAME_HEIGHT//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (155, 89, 182), 5, cv2.LINE_AA)

        # --- 右下角最近一次打分 ---
        if last_score is not None:
            cv2.rectangle(display_frame, (FRAME_WIDTH-420, FRAME_HEIGHT-200), (FRAME_WIDTH-20, FRAME_HEIGHT-20), COLOR_BLACK, -1)
            cv2.rectangle(display_frame, (FRAME_WIDTH-420, FRAME_HEIGHT-200), (FRAME_WIDTH-20, FRAME_HEIGHT-20), COLOR_WHITE, 2)
            score_color = COLOR_STD if last_score >= 60 else COLOR_WARN
            display_frame = put_chinese_text(display_frame, "📊 最近一次出拳", (FRAME_WIDTH-400, FRAME_HEIGHT-185), font_size=20)
            display_frame = put_chinese_text(display_frame, f"总得分：{last_score:.1f}/80", (FRAME_WIDTH-400, FRAME_HEIGHT-150), font_size=24, color=score_color)
            display_frame = put_chinese_text(display_frame, last_suggestion, (FRAME_WIDTH-400, FRAME_HEIGHT-110), font_size=16)

        # 4. 按键监听
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 退出时打印训练总结
            print("\n✅ 训练结束！")
            if score_history:
                print(f"📊 本次训练统计：")
                print(f"   总出拳次数：{len(score_history)}")
                print(f"   平均得分：{np.mean(score_history):.1f}")
                print(f"   最高得分：{np.max(score_history):.1f}")
            break
        # Tab键切换骨骼叠加开关
        if key == ord('\t'):
            skeleton_overlay_enabled = not skeleton_overlay_enabled
            print(f"📌 骨骼叠加：{'开启' if skeleton_overlay_enabled else '关闭'}")

        # 显示画面
        cv2.imshow("🥊 拳击动作AI训练系统", display_frame)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    main()