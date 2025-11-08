import os
import time
from copy import deepcopy
from threading import Lock
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from utils import video_composition, save_image


class VideoWriter:
    def __init__(self, config):
        # 获取配置参数
        self.fps = config['writer']['fps']
        self.video_save_root = os.path.join(config['static_root'], config['static']['video'])
        self.image_save_root = os.path.join(config['static_root'], config['static']['image'])
        self.imgsize = (config['writer']['width'], config['writer']['height'])
        self.continue_frames_threshold = config['writer']['continue_frames_threshold']
        self.previous_frames_threshold = config['writer']['previous_frames_threshold']

        # 缓存变量
        self.buffer = {}  # 用于提交视频合成的缓冲队列
        self.image_name = {}  # 用于记录合成的视频封面图片名字
        self.video_name = {}  # 用于记录合成的视频名字
        self.continue_frames = {}  # 用于去除检测丢失导致的几帧间隔从而引起的视频合成分段
        self.previous_frames = {}  # 用于记录提前帧
        self.video_writer_status = {}  # 用于记录合成状态 waiting appending

        # 线程参数
        self.lock = Lock()  # 线程锁，确保提交任务和结果处理的线程安全
        self.executor = ThreadPoolExecutor(max_workers=1)

    def safe_video_composition(self, name, frames):
        try:
            video_composition(name, frames)
        except Exception as e:
            print(f"[ERROR] Failed to compose video {name}: {e}")

    def get_writer_result(self, data):
        # 1. 判断是否启用了视频记录
        if not data.camera_info.enable_writer:
            return

        # 2. 获取摄像头ip 并 进行初始化
        cip = data.camera_info.name
        # if cip in self.video_name.keys():
        #     data.writer_info.image_name = self.image_name[cip]
        #     data.writer_info.video_name = self.video_name[cip]
        if cip not in self.previous_frames.keys():
            self.buffer[cip] = []
            self.continue_frames[cip] = 0
            self.previous_frames[cip] = deque(maxlen=self.previous_frames_threshold)
            self.video_writer_status[cip] = "waiting"

        # 3. 处理无行人通过时的情况
        if ((not data.tracker_info.person_list and not data.camera_info.enable_abnormal_writer) or
                (data.camera_info.enable_abnormal_writer and not data.action_info.is_abnormal)):
            if self.video_writer_status[cip] == "appending":
                if self.continue_frames[cip] > self.continue_frames_threshold:
                    with self.lock:
                        self.executor.submit(self.safe_video_composition, self.video_name[cip],
                                             deepcopy(self.buffer[cip]))
                        self.buffer[cip].clear()
                    self.video_writer_status[cip] = "waiting"
                    self.continue_frames[cip] = 0
                else:
                    self.buffer[cip].append(data.frame.copy())
                    self.continue_frames[cip] += 1
            else:
                self.previous_frames[cip].append(data.frame.copy())
            return

        # 3. 处理有行人通过时的情况

        ## 3.1 确定异常类型
        cross_condition, abnormal_condition = False, False
        for p in data.tracker_info.person_list:
            if p.crossed and p.movement in data.camera_info.cross_direction:
                cross_condition = True
                break

        if data.action_info.is_abnormal:
            abnormal_condition = True

        ## 3.2 合成初始化
        if ((data.camera_info.enable_crossed_writer and cross_condition) or (
                data.camera_info.enable_abnormal_writer and abnormal_condition)) and self.video_writer_status[
            cip] == "waiting":
            # 更新状态
            self.video_writer_status[cip] = "appending"

            # 确定视频合成类别
            video_type = "base"
            if cross_condition:
                video_type = "cross"
            if abnormal_condition:
                video_type = "abnormal"

            # 写入封面
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.image_name[cip] = f"{self.image_save_root}/{video_type}-{timestamp}.jpg"
            save_image(self.image_name[cip], data.frame.copy())
            data.writer_info.image_name = self.image_name[cip]

            # 初始化并写入保存视频
            self.video_name[cip] = f"{self.video_save_root}/{video_type}-{timestamp}.mp4"
            data.writer_info.video_name = self.video_name[cip]

            # 将提前帧加入缓存队列
            self.buffer[cip].extend(self.previous_frames[cip])
            self.previous_frames[cip].clear()

        ## 3.3 添加需要合成的视频帧
        if self.video_writer_status[cip] == "waiting":
            self.previous_frames[cip].append(data.frame.copy())
        else:
            self.buffer[cip].append(data.frame.copy())
        self.continue_frames[cip] = 0
