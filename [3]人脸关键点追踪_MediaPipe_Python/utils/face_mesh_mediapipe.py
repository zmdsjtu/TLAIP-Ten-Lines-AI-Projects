import os

import cv2
import mediapipe as mp


class InputData:
    def __init__(self, file=0):
        self.cap = None
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
                img = cv2.imread(self.img_list[self.img_id])
                self.img_id += 1
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


class InitFaceTracker:
    def __init__(self, use_static_mode=False, detect_conf=0.5, track_conf=0.5, faces_num=2):
        self.use_static_mode = use_static_mode
        self.detect_conf = detect_conf
        self.track_conf = track_conf
        self.faces_num = faces_num
        self.face = None
        self.init_network()

    def init_network(self):
        self.face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.use_static_mode,
            max_num_faces=self.faces_num,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("init hand tracking down")

    def run_face_tracking(self, get_nex_img):
        while True:
            img_origin = next(get_nex_img)
            if img_origin is None:
                yield [None, None]
                break
            img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            yield [img_origin, self.face.process(img)]


class ShowResult:
    def __init__(self, waitkey_mode=1):
        self.waitkey = waitkey_mode

    def show_result(self, run_hand_tracking):
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 0))
        while True:
            img, results = next(run_hand_tracking)

            if img is None:
                break
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec
                    )
            cv2.imshow('MediaPipe Face Mesh', img)
            if cv2.waitKey(self.waitkey) & 0xFF == 27:
                break
