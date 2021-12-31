# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2021/12/28 21:08
#   Description : Just run this script
# ============================================================

from utils.holistic_mediapipe import InputData, InitHolisticTracker, ShowResult

# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData("test/erkang.mp4")  # input_data = InputData("test/imgs")
# 初始化全身关键点追踪tracker
pose_track = InitHolisticTracker(use_static_mode=input_data.use_img_list, up_body_only=True)
# 获取图像以及结果的generator
run_pose_result = pose_track.run_face_tracking(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key, up_body_only= pose_track.up_body_only).show_result(run_pose_result)

