# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2021/12/27 21:26
#   Description : Just run this script
# ============================================================

from utils.pose_mediapipe import InputData, InitPoseTracker, ShowResult

# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData("test/Star-Lord.mp4")  # input_data = InputData("test/imgs")
# 初始化人体姿态追踪tracker
pose_track = InitPoseTracker(use_static_mode=input_data.use_img_list, up_body_only=False)
# 获取图像以及结果的generator
run_pose_result = pose_track.run_pose_tracking(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key, pose_track.up_body_only).show_result(run_pose_result)
