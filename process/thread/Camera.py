import cv2
from loguru import logger
from threading import Thread, Event

from data_structure import DataStructure


class Camera(Thread):
    def __init__(self, config, idx, i, info):
        super().__init__()
        self.daemon = True

        self.config = config
        self.info = info

        group_name = list(config['camera']['group'][idx].keys())[0]
        self.basic_info = config['camera']['group'][idx][group_name]['ip'][i]
        self.basic_info['enable_reid'] = config['camera']['group'][idx][group_name]['enable_reid']
        self.basic_info['enable_writer'] = config['camera']['group'][idx][group_name]['enable_writer']
        self.basic_info['device'] = config['camera']['group'][idx][group_name]['device']

        self.name = self.basic_info['name']
        self.port = self.basic_info['port']
        self.rate = self.basic_info['rate']
        self.max_qsize = config['camera']['max_qsize']
        if self.rate is None:
            self.rate = 1
            logger.warning(f"未指定摄像头采样率 默认采样率为1")
        else:
            logger.info(f"摄像头采样率已设置为{self.rate}")

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
            self.capture = cv2.VideoCapture(self.port, cv2.CAP_FFMPEG)
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
                if self.count * self.rate >= 1:
                    data = DataStructure()
                    data.fill_camera_info(self.basic_info)
                    data.frame = frame.copy()
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
