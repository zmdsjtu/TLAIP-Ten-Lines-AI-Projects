# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2021/12/28 21:08
#   Description : Just run this script
# ============================================================

from utils.objectron_mediapipe import InputData, InitObjectron, ShowResult

model_name = "Shoe" # "Cup" "Chair" "Camera"
# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData("test/Shoe.mp4")  # input_data = InputData("test/imgs")
detect_3d_object = InitObjectron(object_name="Shoe", static_mode=input_data.use_img_list)
# 获取图像以及结果的generator
run_pose_result = detect_3d_object.run_objectron(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key).show_result(run_pose_result)
