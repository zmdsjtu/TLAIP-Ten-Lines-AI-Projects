# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2021/12/28 21:08
#   Description : Just run this script
# ============================================================

from utils.segmentation_mediapipe import InputData, InitSegmentation, ShowResult

# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData("test/hand_tracking.mp4")  # input_data = InputData("test/imgs")
bg_data = InputData("test/imgs/background", repeat=True, repeat_step= 15)
selfie_segmentation = InitSegmentation(model=0)
# 获取图像以及结果的generator
run_pose_result = selfie_segmentation.run_segmentation(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key).show_result(run_pose_result, bg_data.get_next_img())