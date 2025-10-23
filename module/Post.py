import time
import requests
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading


def post_data(url, result):
    try:
        response = requests.post(url, json=result, timeout=1)
        response.raise_for_status()
        logger.info(f"请求成功\nurl:{url}\nparams:{result}")
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败\nurl:{url}\nparams:{result}\nerror:{str(e)}")


class Post:
    def __init__(self, config):
        self.config = config
        self.urls = config['url']
        self.linger_threshold = config['people tracker']['max_duration_time']
        self.tracking_records = {}

        # 创建线程池
        self.futures = []  # 存储Future对象
        self.lock = threading.Lock()  # 用于线程安全地访问共享数据
        self.executor = ThreadPoolExecutor(max_workers=1)

    def delete_drop_ids(self, data):
        # 删除弃用ID
        if data.camera_info.enable_reid:
            drop_ids = data.reid_info.reid_drop_ids
        else:
            drop_ids = data.tracker_info.drop_ids
        for tid in drop_ids:
            if tid in self.tracking_records:
                del self.tracking_records[tid]

    def info_pack(self, data):
        if not data.camera_info.enable_face_record and not data.camera_info.enable_face_recognition:
            mem_list = data.get_tracker_memory_data()
        else:
            mem_list = data.get_face_memory_data()
        for m in mem_list:
            # 1. 初始化新检测到的目标
            person_id = m["person_id"]
            if person_id not in self.tracking_records:
                self.tracking_records[person_id] = m
                self.tracking_records[person_id]['filter'] = False
                continue

            # 2. 验证数据状态
            del self.tracking_records[person_id]['filter']
            if self.tracking_records[person_id] == m:
                self.tracking_records[person_id]['filter'] = True
            else:
                self.tracking_records[person_id] = m
                self.tracking_records[person_id]['filter'] = False

    def post_data(self, data):
        result_list = []
        if not data.camera_info.enable_face_record and not data.camera_info.enable_face_recognition:
            for _, record in self.tracking_records.items():
                # 1. 过滤不需要的信息
                if record['filter']:
                    continue
                # 2. 过滤没有跨越
                if not record['crossed']:
                    continue

                result_list.append(
                    {
                        "cross": record['crossed'],
                        "direction": record['movement'],
                        "is_linger": True if time.time() - record['create_time'] > self.linger_threshold else False,
                    }
                )

            if not result_list:
                return

            result = {
                "ip": data.camera_info.name,
                "list": result_list,
                "videoPath": data.writer_info.video_name,
                "imagePath": data.writer_info.image_name,
            }
            self.executor.submit(post_data, self.urls['customerCrossFile'], [result])
        elif data.camera_info.enable_face_recognition:
            for _, record in self.tracking_records.items():
                # 1.过滤不需要的信息 2.过滤未被确认的
                if record['filter'] or not record['is_verified']:
                    continue
                result_list.append({"customerId": record["person_name"]})

            if not result_list:
                return

            result = {
                "ip": data.camera_info.name,
                "list": result_list,
            }
            self.executor.submit(post_data, self.urls['customerFaceFile'], [result])
        elif data.camera_info.enable_face_record:
            for _, record in self.tracking_records.items():
                # 1.过滤不需要的信息 2.过滤未被确认的 3.不是新录入的
                if record['filter'] or not record['is_verified'] or not record['is_new']:
                    continue
                item = {
                    "customerId": record["person_name"],
                    "imagePath": record['image_path']
                }
                # 2. 将标志位new标记为False
                record['is_new'] = False
                result_list.append(item)

            if not result_list:
                return

            result = {
                "ip": data.camera_info.name,
                "list": result_list,
            }
            self.executor.submit(post_data, self.urls['customerFaceFile'], [result])

    def post(self, data):
        # 没有行人信息，直接返回
        if not data.tracker_info.person_list:
            return

        self.delete_drop_ids(data)
        self.info_pack(data)
        self.post_data(data)
        del data
