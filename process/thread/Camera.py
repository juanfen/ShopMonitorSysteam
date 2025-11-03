import cv2
from loguru import logger
from copy import deepcopy
from collections import deque
from threading import Thread, Event

from data_structure import CameraInfo, DataStructure


class Camera(Thread):
    def __init__(self, config, idx, i, info):
        super().__init__()
        # 1. 配置线程模式
        self.daemon = True

        # 2. 获取配置 通信队列
        self.info = info
        self.config = config

        # 3. 格式化数据结构
        self.camera_info = CameraInfo()
        group_name = list(config['camera']['group'][idx].keys())[0]
        basic_info = config['camera']['group'][idx][group_name]['ip'][i]
        self.camera_info.k = basic_info['k']
        self.camera_info.rate = basic_info['rate']
        self.camera_info.name = basic_info['name']
        self.camera_info.port = basic_info['port']
        self.camera_info.device = config['camera']['group'][idx][group_name]['device']
        self.camera_info.enable_reid = config['camera']['group'][idx][group_name]['enable_reid']
        self.camera_info.enable_writer = config['camera']['group'][idx][group_name]['enable_writer']
        self.camera_info.enable_crossed_writer = basic_info['enable_crossed_writer']
        self.camera_info.enable_abnormal_writer = basic_info['enable_abnormal_writer']
        self.camera_info.cross_direction = basic_info['cross_direction']
        self.camera_info.enable_face_record = basic_info['enable_face_record']
        self.camera_info.enable_face_recognition = basic_info['enable_face_recognition']
        self.camera_info.enable_silent_detected = basic_info['enable_silent_detected']
        self.camera_info.enable_action_recognition = basic_info['enable_action_recognition']

        # 4. 校验运行参数
        self.max_qsize = config['camera']['max_qsize']
        self.clip_length = config['action recognition']['clip_length']
        if self.camera_info.rate is None:
            self.camera_info.rate = 1
            logger.warning(f"未指定摄像头采样率 默认采样率为1")
        else:
            logger.info(f"摄像头采样率已设置为{self.camera_info.rate}")

        # 5. 配置缓冲参数
        self.clip = deque(maxlen=self.clip_length)
        self.count = 0
        self.capture = None
        self.camera_open = False
        self.finish_event = Event()

    def safe_put(self, data):
        if self.info.qsize() > self.max_qsize:
            self.info.get()
        self.info.put(data)

    def connect_camera(self):
        try:
            if self.capture:
                self.capture.release()
                self.capture = None
            self.capture = cv2.VideoCapture(self.camera_info.port, cv2.CAP_FFMPEG)
            self.camera_open = True
            if not self.capture.isOpened():
                logger.error(f"无法打开相机: {self.name}")
                self.capture.release()
                self.capture = None
                self.camera_open = False
            else:
                logger.info(f"成功打开相机: {self.name}")
        except Exception as e:
            logger.error(f"初始化相机异常: {e}")
            if self.capture:
                self.capture.release()
                self.capture = None
            self.camera_open = False

    def read_camera(self):
        try:
            success, frame = self.capture.read()
            if success:
                self.count += 1
                if self.count * self.camera_info.rate >= 1:
                    data = DataStructure()
                    data.camera_info = deepcopy(self.camera_info)
                    data.frame = frame.copy()
                    self.clip.append(frame.copy())
                    data.clip = list(self.clip)
                    self.safe_put(data)
                    self.count = 0
            else:
                logger.error(f"读取帧失败: {self.name}")
                self.camera_open = False
        except Exception as e:
            logger.error(f"读取异常: {e}")
            self.camera_open = False

    def stop(self):
        self.finish_event.set()
        logger.info(f"相机{self.name} 线程正在退出")

    def run(self):
        self.finish_event.clear()
        while not self.finish_event.is_set():
            while not self.camera_open:
                self.connect_camera()
            self.read_camera()
        logger.info(f"相机{self.name} 线程结束")
