import cv2
import numpy as np


class FrameDifference:
    def __init__(self, config):
        self.diff_threshold = config['camera']['sensitivity']
        self.pre_frame_dict = {}

    def calculate_difference(self, data):
        # 1. 获取当前灰度图
        gray_frame = cv2.cvtColor(data.frame, cv2.COLOR_BGR2GRAY)

        # 2. 获取记录灰度图
        if data.camera_info.name not in self.pre_frame_dict.keys():
            self.pre_frame_dict[data.camera_info.name] = gray_frame
            return 1

        # 3. 对比图片大小是否相同
        pre_frame = self.pre_frame_dict[data.camera_info.name]
        if pre_frame.shape != gray_frame.shape:
            return 1

        # 4. 计算图像差分 与 变化率
        diff = cv2.absdiff(pre_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # 二值化阈值设为30
        change_ratio = np.sum(thresh > 0) / (gray_frame.shape[0] * gray_frame.shape[1])

        # 5. 更新记录图像
        self.pre_frame_dict[data.camera_info.name] = gray_frame
        return change_ratio

    def get_frame_diff_result(self, data, person_lock):
        enable_silent_detected = data.camera_info.enable_silent_detected
        if enable_silent_detected and not person_lock:
            change_ratio = self.calculate_difference(data)
            if change_ratio < self.diff_threshold:
                return True
        return False
