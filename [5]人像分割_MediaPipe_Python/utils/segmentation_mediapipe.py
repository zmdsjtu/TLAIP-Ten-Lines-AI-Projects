import os

import cv2
import mediapipe as mp
import numpy as np


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


class InitSegmentation:
    def __init__(self, model=0):
        self.segmentation = None
        self.model = model
        self.init_network()

    def init_network(self):
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=self.model)

    def run_segmentation(self, get_nex_img):
        while True:
            img_origin = next(get_nex_img)
            if img_origin is None:
                yield [None, None]
                break
            img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            yield [img_origin, self.segmentation.process(img)]


class ShowResult:
    def __init__(self, waitkey_mode=1):
        self.waitkey = waitkey_mode

    def show_result(self, run_hand_tracking, bg_gen):

        while True:
            img, results = next(run_hand_tracking)
            if img is None:
                break
            bg_img = next(bg_gen)
            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            bg_image = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
            output_image = np.where(condition, img, bg_image)
            cv2.imshow('MediaPipe-Segmentation', output_image)
            if cv2.waitKey(self.waitkey) & 0xFF == 27:
                break
