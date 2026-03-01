from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

# 加载开源文生动作模型，自动下载，国内网络无压力
motion_gen_pipe = pipeline(Tasks.text_to_motion, model='modelscope/MotionDiffuse-text2motion')

# 精准prompt生成标准拳击动作，可按需修改
prompt = "一个成年男性做标准的右直拳拳击动作，全身运动，转髋核心发力，完整完成出拳和收拳动作，动作时长3秒，帧率30fps"
# 生成动作
result = motion_gen_pipe(prompt)
motion_data = result['motion']  # 生成的3D人体动作时序数据

# 保存标准动作模板
np.save("../standard_action/standard_straight_generated.npy", motion_data)
print(f"✅ 标准拳击动作已生成，共{motion_data.shape[0]}帧，已保存至standard_action文件夹")