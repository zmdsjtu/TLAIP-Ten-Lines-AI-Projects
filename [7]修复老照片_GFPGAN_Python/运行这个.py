# ============================================================
#   File        : 运行这个.py
#   Author      : zmdsjtu@163.com
#   Created date: 2022/2/16 19:56
#   Description : Just run this script
# ============================================================

from utils.repair_old_photo_gfpgan import InputData, InitGfpgan, ShowResult

# 初始化输入源, file支持数字（相机）以及视频文件路径，图片路径或文件夹路径
input_data = InputData(file = "test/whole_imgs")
# 初始化网络
gfpgan = InitGfpgan()
# 获取图像以及结果的generator
run_pose_result = gfpgan.run_enhance(input_data.get_next_img())
# 显示结果， ESC退出，图片模式按任意键继续
ShowResult(input_data.wait_key).show_result(run_pose_result)