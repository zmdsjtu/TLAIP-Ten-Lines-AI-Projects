# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2021/12/22 18:51
#   Description : Just run this script
# ============================================================

from utils.hand_tracking_mediapipe import InputData, InitHandTracking, ShowResult

# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData(file="test/hand_tracking.mp4") # InputData(0) #默认调用相机  # InputData(file="test/imgs")
InputData(file="test/imgs")
# 初始化手势追踪tracker
hand_track = InitHandTracking(use_static_mode = input_data.use_img_list)
# 获取图像以及结果的generator
run_hand_tracking_result = hand_track.run_hand_tracking(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key).show_result(run_hand_tracking_result)
