import torch
import random
import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
from torchreid.reid.utils import FeatureExtractor

from data_structure import ReidPersonInfo


class ReID:
    def __init__(self, config):
        self.config = config
        self.extractor = FeatureExtractor(
            model_name='osnet_x0_25',
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        # 全局ID计数器
        self.next_global_id = 0

        # 新增：全局ID到固定颜色的映射
        self.id_color_map = {}

        # person_cam_map: 摄像头跟踪ID -> 全局ID ("cam1-123" -> 456)
        self.person_cam_map = {}

        # global_id_map: 全局ID -> 特征向量队列 (存储最近N个特征向量)
        self.global_id_map = defaultdict(lambda: deque(maxlen=self.config['reid']['max_features']))

    def _get_color_for_id(self, global_id):
        """为全局ID分配固定颜色"""
        if global_id not in self.id_color_map:
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self.id_color_map[global_id] = color
            logger.info(f"为全局ID {global_id} 分配新颜色: {color}")
        return self.id_color_map[global_id]

    def delete_drop_ids(self, data):
        """清理需要丢弃的ID"""
        cam_name = data.camera_info.name
        for track_id in data.tracker_info.drop_ids:
            cam_track_id = f"{cam_name}-{track_id}"
            if cam_track_id in self.person_cam_map:
                global_id = self.person_cam_map[cam_track_id]
                if global_id not in data.reid_info.reid_drop_ids:
                    data.reid_info.reid_drop_ids.append(global_id)
                del self.person_cam_map[cam_track_id]
                if global_id in self.global_id_map.keys():
                    del self.global_id_map[global_id]
                logger.info(f"Cleaned up ID: {cam_track_id} (Global ID: {global_id})")

    def get_reid_result(self, data):
        # 1. 判断是否启用了reid
        if not data.camera_info.enable_reid:
            for p in data.tracker_info.person_list:
                rp = ReidPersonInfo()
                rp.person_reid = p.person_id
                data.reid_info.person_list.append(rp)
            return

        # 2. 去除过期的reid
        self.delete_drop_ids(data)

        # 3. 进行重识别
        cam_name = data.camera_info.name
        for p in data.tracker_info.person_list:
            # 1. 创建reid的数据项
            rp = ReidPersonInfo()

            # 2. 提取外观特征
            rp.embed = self.extractor(p.person_roi).cpu().numpy()

            # 3. 根据特征码识别是否已存在映射 如有直接更新特征
            cam_track_id = f"{cam_name}-{p.person_id}"
            if cam_track_id in self.person_cam_map:
                global_id = self.person_cam_map[cam_track_id]
                self.global_id_map[global_id].append(rp.embed)
                rp.person_reid = global_id
                p.color = self._get_color_for_id(global_id)
                continue

            # 4. 若是新目标 尝试匹配现有全局ID
            min_distance = float('inf')
            best_global_id = None
            for global_id, features in self.global_id_map.items():
                # 计算平均特征向量
                if not features:
                    continue
                avg_feature = np.mean(features, axis=0)

                # 计算距离
                distance = cosine(rp.embed.flatten(), avg_feature.flatten())

                # 更新最佳匹配
                if distance < self.config['reid']['threshold'] and distance < min_distance:
                    min_distance = distance
                    best_global_id = global_id

            # 5. 处理匹配结果
            if best_global_id is not None:
                self.person_cam_map[cam_track_id] = best_global_id
                self.global_id_map[best_global_id].append(rp.embed)
                rp.person_reid = best_global_id
                p.color = self._get_color_for_id(best_global_id)
                logger.info(
                    f"Matched {cam_track_id} to existing Global ID: {best_global_id} (Distance: {min_distance:.4f})")
            else:
                self.next_global_id += 1
                new_global_id = self.next_global_id
                self.person_cam_map[cam_track_id] = new_global_id
                self.global_id_map[new_global_id] = deque([rp.embed], maxlen=self.config['reid']['max_features'])
                rp.person_reid = new_global_id
                p.color = self._get_color_for_id(new_global_id)
                logger.info(f"Created new Global ID: {new_global_id} for {cam_track_id}")

            # 6. 记录新数据
            data.reid_info.person_list.append(rp)
