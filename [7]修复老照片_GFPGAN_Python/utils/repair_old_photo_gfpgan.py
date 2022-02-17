import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer


class InputData:
    def __init__(self, file=0, repeat=False, repeat_step=1):
        self.cap = None
        self.repeat = repeat
        self.repeat_step = repeat_step
        self.img_list = []
        self.img_id = 0
        self.img_type_list = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'webp'}
        self.deal_with_input(file)
        self.use_img_list = len(self.img_list) > 0
        self.wait_key = 0 if self.use_img_list else 1
        self.use_static_mode = self.use_img_list

    def gen_img_list(self, path):
        for item in os.listdir(path):
            if item.split(".")[-1] in self.img_type_list:
                self.img_list.append(os.path.join(path, item))

    def deal_with_input(self, file):
        path_valid = False
        if isinstance(file, int):
            # use camera
            self.cap = cv2.VideoCapture(file)
            path_valid = True
        elif isinstance(file, str):
            if os.path.isdir(file):
                # use img list
                self.gen_img_list(file)
                if len(self.img_list) > 0:
                    path_valid = True
                else:
                    print("no images in", file)
            elif os.path.isfile(file):
                if file.split(".")[-1] in self.img_type_list:
                    # only one image
                    self.img_list.append(file)
                    path_valid = True
                else:
                    self.cap = cv2.VideoCapture(file)
                    if self.cap.isOpened():
                        print("video path is", file)
                        path_valid = True
                    else:
                        print("video path is not valid, path is:", file)
        if not path_valid:
            print("Invalid input! Use camera 0 instead!")
            self.cap = cv2.VideoCapture(0)

    def get_next_img(self):
        if self.use_img_list:
            while self.img_id < len(self.img_list):
                img = cv2.imread(self.img_list[int(self.img_id)])
                self.img_id += 1 / self.repeat_step
                if self.img_id >= len(self.img_list) and self.repeat:
                    self.img_id = 0
                if img is not None:
                    yield img
            yield None
        else:
            while True:
                _, img = self.cap.read()
                if img is None:
                    yield None
                    break
                yield img

    def release_cap(self):
        if self.cap:
            self.cap.release()


class InitGfpgan:
    def __init__(self, repair_face=False, save_image=False):
        self.restorer = None
        self.init_network()

    def init_network(self):
        # background upsampler
        bg_upsampler = None
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode

        # set up GFPGAN restorer
        self.restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

    def run_enhance(self, get_nex_img):
        while True:
            img_origin = next(get_nex_img)
            if img_origin is None:
                yield [None, None]
                break
            yield [img_origin, self.restorer.enhance(img_origin)]


class ShowResult:
    def __init__(self, waitkey_mode=1):
        self.waitkey = waitkey_mode

    def show_result(self, run_hand_tracking):
        while True:
            img, results = next(run_hand_tracking)
            if img is None:
                break
            cropped_faces, restored_faces, restored_img = results
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                cv2.imshow("face_" + str(idx), cmp_img)

            # save restored img
            if restored_img is not None:
                restored_img_small = cv2.resize(restored_img, (int(img.shape[1]), int(img.shape[0])))
                cmp_img_all = np.concatenate((img, restored_img_small), axis=1)
                cv2.imshow("result", cmp_img_all)
            if cv2.waitKey(self.waitkey) & 0xFF == 27:
                break
