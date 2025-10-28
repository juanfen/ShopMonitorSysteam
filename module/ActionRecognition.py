import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from loguru import logger
from copy import deepcopy
from typing import List, Dict
from collections import deque

from data_structure import ActionPersonInfo


class ActionRecognition:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_config = config.get('action recognition', {})

        # 模型参数
        self.model_path = self.action_config.get('model_path', '')
        self.num_classes = self.action_config.get('num_classes', 4)
        self.threshold = self.action_config.get('threshold', 0.6)
        self.clip_length = self.action_config.get('clip_length', 32)
        self.input_size = self.action_config.get('input_size', 224)
        self.prediction_interval = self.action_config.get('prediction_interval', 0.5)

        # 动作类别映射
        self.action_classes = ['drinking', 'eating', 'hiding', 'tearing']
        self.abnormal_actions = self.action_config.get('abnormal_actions', ['hiding', 'tearing'])

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 帧缓冲区：为每个person_reid维护一个帧序列
        self.frame_buffers = {}  # {person_reid: deque}
        self.max_buffer_size = self.clip_length * 2

        # 预测状态记录
        self.prediction_records = {}  # {person_reid: last_prediction_time}

        # 加载模型
        self.model = None
        self._load_model()

        # 预处理配置
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        logger.info(f"动作识别模块初始化完成，设备: {self.device}, 阈值: {self.threshold}")

    def _load_model(self):
        """加载SlowFast模型"""
        try:
            from pytorchvideo.models.hub import slowfast_r50

            # 创建模型结构
            self.model = SlowFastModel(self.num_classes, pretrained=False)

            if self.model_path and os.path.exists(self.model_path):
                # 加载训练好的权重
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # 处理检查点
                state_dict = self._process_checkpoint(checkpoint)

                # 加载权重
                self.model.load_state_dict(state_dict)
                logger.info(f"动作识别模型加载成功: {self.model_path}")
            else:
                # 使用预训练模型
                self.model = SlowFastModel(self.num_classes, pretrained=True)
                logger.info("使用预训练SlowFast模型")

            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"动作识别模型加载失败: {e}")
            self.model = None

    def _process_checkpoint(self, checkpoint):
        """处理检查点文件 - 与你的demo保持一致"""
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

        # 移除module前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v

        return new_state_dict

    def preprocess_frames(self, frames: List[np.ndarray]) -> tuple:
        """预处理帧序列用于SlowFast模型"""
        if len(frames) == 0:
            raise ValueError("输入帧列表为空")

        # 如果帧数不足，重复最后一帧
        if len(frames) < self.clip_length:
            frames = frames + [frames[-1]] * (self.clip_length - len(frames))

        # 采样策略：从视频中均匀采样clip_length帧
        total_frames = len(frames)
        if total_frames > self.clip_length:
            indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # 转换为tensor并预处理
        frame_tensors = []
        for frame in sampled_frames:
            try:
                # 确保帧是RGB格式且数据类型正确
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # 调整尺寸
                    frame_resized = cv2.resize(frame, (256, 256))

                    # 转换为tensor并调整维度 [H, W, C] -> [C, H, W]
                    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()

                    # 应用与训练时相同的预处理
                    frame_tensor = frame_tensor / 255.0
                    frame_tensor = T.functional.center_crop(frame_tensor, [224, 224])
                    frame_tensor = T.functional.normalize(frame_tensor,
                                                          mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

                    frame_tensors.append(frame_tensor)
            except Exception as e:
                logger.warning(f"帧处理失败: {e}")
                continue

        if len(frame_tensors) < self.clip_length:
            raise ValueError(f"有效帧数量不足: {len(frame_tensors)}/{self.clip_length}")

        # 堆叠帧 [T, C, H, W]
        frames_tensor = torch.stack(frame_tensors, dim=0)

        # 创建SlowFast双路径输入
        slow_path = frames_tensor[::4].unsqueeze(0)  # [1, 8, C, H, W]
        fast_path = frames_tensor.unsqueeze(0)  # [1, 32, C, H, W]

        # 调整维度 [B, C, T, H, W]
        slow_path = slow_path.transpose(1, 2).contiguous()  # [1, C, 8, H, W]
        fast_path = fast_path.transpose(1, 2).contiguous()  # [1, C, 32, H, W]

        return slow_path, fast_path

    def predict_action(self, frames: List[np.ndarray]) -> Dict:
        """对帧序列进行动作识别预测"""
        if self.model is None or len(frames) < self.clip_length:
            return {
                'success': False,
                'action_label': 'None',
                'confidence': 0.0,
                'is_abnormal': False
            }

        try:
            with torch.no_grad():
                # 预处理帧
                slow_input, fast_input = self.preprocess_frames(frames)
                slow_input = slow_input.to(self.device)
                fast_input = fast_input.to(self.device)

                # 模型预测
                outputs = self.model([slow_input, fast_input])
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                # 获取所有类别的概率用于调试
                all_probabilities = probabilities[0].cpu().numpy()

                # 应用置信度阈值
                if confidence < self.threshold:
                    action_label = "None"
                    is_abnormal = False
                else:
                    action_label = self.action_classes[predicted_class]
                    is_abnormal = action_label in self.abnormal_actions

                # 调试模式下输出详细预测信息
                if self.config.get('enable_debug', False):
                    # 构建概率详情字符串
                    prob_details = []
                    for i, prob in enumerate(all_probabilities):
                        prob_details.append(f"{self.action_classes[i]}: {prob:.4f}")
                    prob_str = ", ".join(prob_details)

                    logger.debug(f"动作识别预测结果: {action_label} (置信度: {confidence:.4f})")
                    logger.debug(f"所有类别概率: [{prob_str}]")
                    logger.debug(f"是否异常行为: {is_abnormal}")
                    logger.debug(f"阈值: {self.threshold}, 预测类别索引: {predicted_class}")

                return {
                    'success': True,
                    'action_label': action_label,
                    'confidence': confidence,
                    'predicted_class': predicted_class,
                    'is_abnormal': is_abnormal,
                    'timestamp': time.time(),
                    'all_probabilities': all_probabilities.tolist()  # 可选：返回所有概率
                }

        except Exception as e:
            logger.error(f"动作识别预测失败: {e}")
            return {
                'success': False,
                'action_label': 'None',
                'confidence': 0.0,
                'is_abnormal': False,
                'error': str(e)
            }

    def update_frame_buffer(self, person_reid: int, person_roi: np.ndarray):
        """更新指定人员的帧缓冲区"""
        if person_reid not in self.frame_buffers:
            self.frame_buffers[person_reid] = deque(maxlen=self.max_buffer_size)

        # 将ROI区域添加到缓冲区
        self.frame_buffers[person_reid].append(person_roi.copy())

    def should_predict(self, person_reid: int) -> bool:
        """判断是否应该对指定人员进行预测"""
        current_time = time.time()

        if person_reid not in self.prediction_records:
            self.prediction_records[person_reid] = 0

        # 检查预测间隔
        if current_time - self.prediction_records[person_reid] < self.prediction_interval:
            return False

        # 检查缓冲区是否有足够帧数
        if person_reid not in self.frame_buffers or len(self.frame_buffers[person_reid]) < self.clip_length:
            return False

        return True

    def delete_drop_ids(self, data):
        """删除不再跟踪的ID的缓冲区"""
        if data.camera_info.enable_reid:
            drop_ids = data.reid_info.reid_drop_ids
        else:
            drop_ids = data.tracker_info.drop_ids

        for tid in drop_ids:
            if tid in self.frame_buffers:
                del self.frame_buffers[tid]
            if tid in self.prediction_records:
                del self.prediction_records[tid]

    def get_action_result(self, data):
        """主处理函数 - 集成到现有流水线"""
        start_time = time.time()

        # 清空上一次的结果
        data.action_info.person_list = []
        data.action_info.abnormal_count = 0

        # 如果没有启用动作识别，直接返回空结果
        if not getattr(data.camera_info, 'enable_action_recognition', False):
            for _ in data.tracker_info.person_list:
                data.action_info.person_list.append(ActionPersonInfo())
            return

        # 删除不再跟踪的ID
        self.delete_drop_ids(data)

        # 统计变量
        prediction_count = 0
        action_results = {}

        # 处理每个行人
        for p, rp in zip(data.tracker_info.person_list, data.reid_info.person_list):
            person_reid = rp.person_reid

            # 创建动作信息对象
            action_person = ActionPersonInfo()

            # 更新帧缓冲区
            self.update_frame_buffer(person_reid, p.person_roi)

            # 初始化result变量
            result = None

            # 检查是否应该进行预测
            if self.should_predict(person_reid):
                # 获取最近的帧序列
                frames_list = list(self.frame_buffers[person_reid])[-self.clip_length:]

                # 进行动作识别
                result = self.predict_action(frames_list)

                # 更新预测记录时间
                self.prediction_records[person_reid] = time.time()

                # 填充动作信息
                action_person.action_label = result['action_label']
                action_person.action_confidence = result['confidence']
                action_person.is_abnormal = result.get('is_abnormal', False)
                action_person.action_roi = p.person_roi

                # 统计预测结果
                if result['success'] and result['action_label'] != 'None':
                    prediction_count += 1
                    action_label = result['action_label']
                    if action_label in action_results:
                        action_results[action_label] += 1
                    else:
                        action_results[action_label] = 1

                # 调试模式下输出个体预测结果 - 移到if语句块内部
                if self.config['enable_debug'] and result is not None and result['success']:
                    logger.debug(f"人员 {person_reid} 动作识别: {result['action_label']} "
                                 f"(置信度: {result['confidence']:.4f}, "
                                 f"异常: {result.get('is_abnormal', False)})")
            else:
                # 如果没有进行预测，保持默认值
                action_person.action_label = "None"
                action_person.action_confidence = 0.0
                action_person.is_abnormal = False
                action_person.action_roi = p.person_roi

                # 调试模式下输出未预测的信息
                if self.config['enable_debug']:
                    buffer_size = len(self.frame_buffers.get(person_reid, []))
                    logger.debug(f"人员 {person_reid} 未进行预测 - 缓冲区大小: {buffer_size}/{self.clip_length}")

            # 统计异常行为
            if action_person.is_abnormal:
                data.action_info.abnormal_count += 1

            data.action_info.person_list.append(action_person)

        # 调试信息
        if self.config['enable_debug']:
            processing_time = time.time() - start_time
            active_buffers = len([buf for buf in self.frame_buffers.values() if len(buf) >= self.clip_length])

            # 构建详细的调试信息
            debug_info = f"动作识别时间: {processing_time:.3f}s, 有效缓冲区: {active_buffers}"

            if prediction_count > 0:
                debug_info += f", 预测次数: {prediction_count}"
                if action_results:
                    action_details = ", ".join([f"{action}:{count}" for action, count in action_results.items()])
                    debug_info += f", 动作分布: [{action_details}]"

            if data.action_info.abnormal_count > 0:
                debug_info += f", 异常行为: {data.action_info.abnormal_count}"

            logger.info(debug_info)


class SlowFastModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        try:
            from pytorchvideo.models.hub import slowfast_r50
            self.model = slowfast_r50(pretrained=pretrained)

            # 替换分类头
            in_features = self.model.blocks[6].proj.in_features
            self.model.blocks[6].proj = nn.Linear(in_features, num_classes)

        except Exception as e:
            logger.error(f"SlowFast模型初始化失败: {e}")
            raise e

    def forward(self, x):
        return self.model(x)