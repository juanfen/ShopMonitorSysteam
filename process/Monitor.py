import os
import cv2
import time
import torch
import numpy as np
from queue import Queue
from loguru import logger

from module import PeopleTracker, FaceRecognition, ReID, FrameDifference, VideoWriter, Post, ActionRecognition
from .thread import Camera
from utils import set_logger


def plot_info(data):
    k = data.camera_info.k
    for p, rp, fp, ap in zip(data.tracker_info.person_list, data.reid_info.person_list, data.face_info.person_list, data.action_info.person_list):
        name = fp.person_name
        x1, y1, x2, y2 = p.xyxy
        color = p.color
        cv2.rectangle(data.frame, (x1, y1), (x2, y2), color, 5)

        person_reid = rp.person_reid
        action_text = f"{ap.action_label}({ap.action_confidence:.2f})" if ap.action_label != "None" else ""
        label = f"ID:{person_reid} {name} {action_text}"
        cv2.putText(data.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)

        track = p.track
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(data.frame, [points], isClosed=False, color=(0, 255, 0), thickness=5)

    # 显示异常行为计数
    if hasattr(data.action_info, 'abnormal_count') and data.action_info.abnormal_count > 0:
        cv2.putText(data.frame, f"Abnormal Actions: {data.action_info.abnormal_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(data.frame, (k, 0), (k, data.frame.shape[0] - 1), (0, 0, 255), 10)


def monitor(config, g_idx, info):
    # 1. 获取组名 摄像头数量
    group_name = list(config['camera']['group'][g_idx].keys())[0]
    camera_nums = len(config['camera']['group'][g_idx][group_name]['ip'])

    # 2. 为当前进程分配GPU
    if config['camera']['group'][g_idx][group_name]['device'].lower() != 'cpu':
        device_id = int(config['camera']['group'][g_idx][group_name]['device'][-1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        torch.cuda.set_device(0)

    # 3. 设置日志
    log_name = os.path.join(config['static_root'], config['static']['log'], group_name + '.log')
    set_logger(log_name, config['log']['format'])

    # 4. 初始化摄像头
    camera_to_main = Queue()
    camera_list = [Camera(config, g_idx, i, camera_to_main) for i in range(camera_nums)]

    # 5. 初始化 帧间差分 跟踪 重识别 人脸识别 模块
    frame_diff = FrameDifference(config)
    tracker_dict = {item['name']: PeopleTracker(config) for item in config['camera']['group'][g_idx][group_name]['ip']}
    reid = ReID(config)
    recognizer = FaceRecognition(config)
    action_recognizer = ActionRecognition(config)  # 新增
    writer = VideoWriter(config)
    post = Post(config)

    # 6. 启动所有线程
    for i in range(camera_nums):
        camera_list[i].start()

    # 7. 初始化缓冲参数
    fps_buffer = []
    person_lock = False
    while True:
        data = camera_to_main.get()

        # 1. 帧间差分检测
        frame_diff_res = frame_diff.get_frame_diff_result(data, person_lock)
        if frame_diff_res:
            fps_buffer.clear()
            if config['enable_ui'] and info.qsize() < config['ui']['max_qsize']:
                info.put({"camera_ip": data.camera_info.name, "frame": data.frame})
            continue

        # 2. 行人跟踪 重识别 人脸识别
        tracker_dict[data.camera_info.name].get_tracker_result(data)
        person_lock = data.tracker_info.person_lock
        reid.get_reid_result(data)
        recognizer.get_recognizer_result(data)
        action_recognizer.get_action_result(data)  # 新增

        # 3. 绘制结果
        plot_info(data)

        # 4. 视频合成 结果提交
        writer.get_writer_result(data)
        post.post(data)

        # 5. 帧率统计
        if config['enable_debug']:
            fps_buffer.append(time.time())
            if len(fps_buffer) > 25:
                fps_buffer.pop(0)
            if len(fps_buffer) > 1:
                fps = len(fps_buffer) / (fps_buffer[-1] - fps_buffer[0])
                logger.info(f"FPS: {fps:.1f}")

        # 6. 向主进程通信
        if config['enable_ui'] and info.qsize() < config['ui']['max_qsize']:
            info.put({"camera_ip": data.camera_info.name, "frame": data.frame})
