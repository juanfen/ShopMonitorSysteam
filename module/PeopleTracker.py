import os
import time
from loguru import logger
from ultralytics import YOLO
from collections import defaultdict

from utils import check_movement, check_gathering_by_clustering
from data_structure import PersonInfo


class PeopleTracker:
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化Yolo
        path = os.path.join(config['static_root'], config['static']['model'], config['people tracker']['model_path'])
        self.people_tracker = YOLO(path)

        # 初始化缓存参数
        self.person_count = 0  # 行人丢失帧数记录
        self.track_history = defaultdict(lambda: [])
        self.missing_frames = defaultdict(int)  # 每个ID连续未出现的帧数
        self.create_time = defaultdict(float)

    def tracker_update(self, data):
        result = self.people_tracker.track(data.frame, verbose=False, classes=[0], conf=0.5, persist=True)[0]

        boxes = result.boxes.xywh.cpu().numpy()

        track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []
        current_ids = set(track_ids)
        previous_ids = set(self.track_history.keys())

        # 重置当前存在的ID的计数器
        for tid in current_ids:
            self.missing_frames[tid] = 0

        # 增加未出现ID的计数器
        for tid in previous_ids - current_ids:
            self.missing_frames[tid] += 1

        # 记录新ID的创建时间
        for tid in current_ids - previous_ids:
            self.create_time[tid] = time.time()

        # 检查需要丢弃的ID
        for tid in list(self.missing_frames.keys()):
            if self.missing_frames[tid] > self.config['people tracker']['max missing frames'] and tid not in current_ids:
                data.tracker_info.drop_ids.append(tid)
                # 清理历史数据
                if tid in self.track_history:
                    del self.track_history[tid]
                if tid in self.missing_frames:
                    del self.missing_frames[tid]
                if tid in self.create_time:
                    del self.create_time[tid]

        return boxes, track_ids

    def info_process(self, data, boxes, track_ids):
        center_list = []
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_list.append((x, y))
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            person_roi = data.frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # 更新轨迹
            track = self.track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > self.config['people tracker']['max tracking frames']:
                track.pop(0)

            check = check_movement(track, data.camera_info.k)

            # 记录被解析的数据
            person_info = PersonInfo()
            person_info.person_id = int(track_id)
            person_info.person_roi = person_roi
            person_info.xyxy = (x1, y1, x2, y2)
            person_info.track = track
            person_info.movement = check[0]
            person_info.crossed = check[1]
            person_info.create_time = self.create_time[track_id]
            data.tracker_info.person_list.append(person_info)

        # 进行聚类判断是否聚众
        flag, _ = check_gathering_by_clustering(center_list)
        data.tracker_info.is_gathering = flag

    def update_person_lock(self, data):
        person_state = True if data.tracker_info.person_list else False
        if person_state:
            person_lock = True
            self.person_count = 0
        elif self.person_count < self.config['silent_detected']['person_lock']:
            person_lock = True
            self.person_count += 1
        else:
            person_lock = False
        return person_lock

    def get_tracker_result(self, data):
        start_time = time.time()

        # 1. 判断是否需要进行行人跟踪
        if data.camera_info.enable_action_recognition:
            return

        boxes, track_ids = self.tracker_update(data)
        self.info_process(data, boxes, track_ids)
        data.tracker_info.person_lock = self.update_person_lock(data)

        if self.config['enable_debug']:
            logger.info("People Tracker Time {}".format(time.time() - start_time))
