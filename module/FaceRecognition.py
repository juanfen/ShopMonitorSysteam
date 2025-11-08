import os
import time
import uuid

import cv2
import torch
import numpy as np
from loguru import logger
from copy import deepcopy
from insightface.app import FaceAnalysis

from data_structure import FacePersonInfo


class FaceRecognition:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fr_thresh = config['face recognition']['threshold']
        self.fr_mqt = config['face recognition']['max_query_times']

        self.records = {}
        self.tracking_records = {}
        self.names = []
        self.features = []
        self.features_matrix = None

        self.path = os.path.join(
            config['static_root'],
            config['static']['face_features']
        )

        self.app = FaceAnalysis(name=config['face recognition']['model'])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))  # 使用 GPU (ctx_id=0)，CPU 则设置为 -1

    def delete_drop_ids(self, data):
        # 删除弃用id
        if data.camera_info.enable_reid:
            drop_ids = data.reid_info.reid_drop_ids
        else:
            drop_ids = data.tracker_info.drop_ids
        for tid in drop_ids:
            if tid in self.tracking_records.keys():
                del self.tracking_records[tid]

    def reload_features(self):
        customer_ids = os.listdir(self.path)
        for customer_id in customer_ids:
            if customer_id not in self.records:
                customer_feature_root = os.path.join(self.path, customer_id)
                # 添加检查：确保路径存在且是目录
                if not os.path.exists(customer_feature_root) or not os.path.isdir(customer_feature_root):
                    continue
                customer_feature_paths = [os.path.join(customer_feature_root, it) for it in os.listdir(customer_feature_root)]
                self.names.extend([customer_id] * len(customer_feature_paths))
                self.features.extend([np.load(p) for p in customer_feature_paths])
        self.features_matrix = np.array(self.features)

    def save_face_features(self, customer_id, current_feat, frame):
        customer_dir = os.path.join(self.path, customer_id)

        # 获取所有特征文件（过滤非特征文件）
        all_files = os.listdir(customer_dir)
        feat_files = [f for f in all_files if f.endswith('_feat.npy')]
        feat_nums = len(feat_files)
        max_nums = self.config['face recognition']['max_face_save_nums']

        # 自动清理旧文件逻辑（当达到或超过阈值时）
        if feat_nums >= max_nums:
            # 获取文件及其修改时间
            file_times = []
            for file in feat_files:
                file_path = os.path.join(customer_dir, file)
                if os.path.isfile(file_path):
                    mod_time = os.path.getmtime(file_path)
                    file_times.append((file, mod_time))

            # 按修改时间排序（旧->新）
            file_times.sort(key=lambda x: x[1])

            # 计算需删除的文件数（保留max_nums-1个位置）
            delete_count = feat_nums - max_nums + 1
            for i in range(delete_count):
                oldest_file, _ = file_times[i]
                oldest_path = os.path.join(customer_dir, oldest_file)
                try:
                    os.remove(oldest_path)
                    logger.info(f"Deleted oldest feature: {oldest_path}")
                except Exception as e:
                    logger.error(f"Delete failed: {oldest_path} - {str(e)}")

        # 保存新特征（使用时间戳命名）
        timestamp = str(int(time.time() * 1000))  # 毫秒级时间戳防重名
        feat_path = os.path.join(customer_dir, f"{timestamp}_feat.npy")
        np.save(feat_path, current_feat)
        logger.info(f"Saved new feature: {feat_path}")
        img_path = os.path.join(self.config['static_root'], self.config['static']['image'], f"{customer_id}_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        logger.info(f"Saved new img: {img_path}")
        return img_path

    def info_process(self, data):
        # 遍历当前行人列表
        for p, rp in zip(data.tracker_info.person_list, data.reid_info.person_list):
            # 1. 状态过滤：已确认身份则跳过
            person_id = rp.person_reid
            if person_id in self.tracking_records:
                print("匹配成功 person_id：{}".format(person_id))
                fp = self.tracking_records[person_id]
                if fp.is_verified:
                    fp.is_new = False
                    data.face_info.person_list.append(deepcopy(fp))
                    continue

            # 2 初始化跟踪记录
            if person_id not in self.tracking_records:
                self.tracking_records[person_id] = FacePersonInfo()
            fp = self.tracking_records[person_id]

            # 3 人脸检测 特征提取
            faces = self.app.get(p.person_roi)
            if len(faces) == 0:
                continue

            # 4 选择最大人脸
            main_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            current_feat = main_face.normed_embedding

            # 5 特征比对
            if self.features_matrix is not None and self.features_matrix.shape[0] != 0:
                similarities = np.dot(self.features_matrix, current_feat)  # 计算所有特征与当前特征的相似度
                best_index = np.argmax(similarities)  # 找到最大相似度的索引
                best_match = (self.names[best_index], similarities[best_index]) if similarities.size > 0 else ('', 0)
            else:
                best_match = ('', 0)

            # 6 更新识别结果
            print("匹配率：",best_match[1])
            if best_match[1] >= self.fr_thresh:
                fp.person_name = best_match[0]
                fp.is_verified = True  # 确认身份后停止更新
                self.save_face_features(best_match[0], current_feat, data.frame)
                fp.is_new = False
            else:
                fp.retry_times += 1
                if self.config['enable_debug']:
                    logger.debug(f"人脸识别重试: 人员ID {person_id}, 次数 {fp.retry_times}")
                print("重试次数",fp.retry_times)
                if fp.retry_times >= self.fr_mqt and data.camera_info.enable_face_record:
                    generated_uuid = str(uuid.uuid4())
                    os.makedirs(os.path.join(self.path, generated_uuid), exist_ok=True)
                    fp.person_name = generated_uuid
                    fp.image_path = self.save_face_features(generated_uuid, current_feat, data.frame)
                    fp.is_verified = True
                    fp.is_new = True
            data.face_info.person_list.append(deepcopy(fp))

    def get_recognizer_result(self, data):
        start_time = time.time()

        # 1. 判断是否需要进行人脸识别
        if not data.camera_info.enable_face_recognition and not data.camera_info.enable_face_record:
            for _ in data.tracker_info.person_list:
                fp = FacePersonInfo()
                data.face_info.person_list.append(fp)
            return

        # 2. 没有行人时直接返回
        if not data.tracker_info.person_list:
            return

        # 3. 删除弃用的id
        self.delete_drop_ids(data)

        # 4. 加载存储的特征
        self.reload_features()

        # 5. 人脸识别
        self.info_process(data)

        # 6.调试
        if self.config['enable_debug']:
            logger.info("Face Recognition Time: {}".format(time.time() - start_time))
