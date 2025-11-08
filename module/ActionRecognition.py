import os
import time
import numpy as np
from copy import deepcopy
from loguru import logger

import torch
from torchvision.transforms import v2

from .net import SlowFast
from data_structure import ActionInfo


class ActionRecognition:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_config = config.get('action recognition', {})

        # 模型参数
        self.model_path = os.path.join(config['static_root'], config['static']['model'],
                                       config['action recognition']['model_path'])
        self.num_classes = self.action_config['num_classes']
        self.threshold = self.action_config['threshold']
        self.clip_length = self.action_config['clip_length']
        self.input_size = self.action_config['input_size']
        self.prediction_interval = self.action_config['prediction_interval']

        # 推理准备
        self.count = 0
        self.model = None
        self.action_info = ActionInfo()
        self.action_classes = ['drinking', 'eating', 'hiding', 'tearing']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.load_model()

    def load_model(self):
        self.model = SlowFast(self.num_classes, pretrained=False)

        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_frames(self, frames):
        np_data = np.asarray(frames)
        total_frames = np_data.shape[0]
        if total_frames < self.clip_length:  # 不足补齐最后一帧
            pad_num = self.clip_length - total_frames
            pad_frames = np.repeat(np_data[-1][np.newaxis, ...], pad_num, axis=0)
            sample_data = np.concatenate([np_data, pad_frames], axis=0)
        else:
            sample_data = np_data

        fast_data = sample_data[np.arange(0, self.clip_length, 2)]
        # print(fast_data.shape)

        fast_frames = self.transform(torch.tensor(fast_data))
        slow_frames = fast_frames[::4].clone()

        slow_path = slow_frames.permute(1, 0, 2, 3)
        fast_path = fast_frames.permute(1, 0, 2, 3)

        return slow_path, fast_path

    def predict_action(self, data):
        with torch.no_grad():
            # 1. 预处理帧
            slow_input, fast_input = self.preprocess_frames(data.clip)
            slow_input = slow_input.unsqueeze(0).to(self.device)
            fast_input = fast_input.unsqueeze(0).to(self.device)

            # 2. 模型预测
            outputs = self.model([slow_input, fast_input])
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            self.action_info.action_confidence = probabilities[0][predicted_class].item()
            all_probabilities = probabilities[0].cpu().numpy()

            # 3. 应用置信度阈值
            if self.action_info.action_confidence < self.threshold:
                self.action_info.action_label = "None"
                self.action_info.is_abnormal = False
            else:
                self.action_info.action_label = self.action_classes[predicted_class]
                self.action_info.is_abnormal = True

            # 4. 调试模式下输出详细预测信息
            if self.config['enable_debug']:
                # 构建概率详情字符串
                prob_details = []
                for i, prob in enumerate(all_probabilities):
                    prob_details.append(f"{self.action_classes[i]}: {prob:.4f}")
                prob_str = ", ".join(prob_details)

                logger.debug(
                    f"动作识别预测结果: {self.action_info.action_label} (置信度: {self.action_info.action_confidence:.4f})")
                logger.debug(f"所有类别概率: [{prob_str}]")
                logger.debug(f"是否异常行为: {self.action_info.is_abnormal}")
                logger.debug(f"阈值: {self.threshold}, 预测类别索引: {predicted_class}")

    def get_action_result(self, data):
        start_time = time.time()

        # 1. 判断是否需要进行动作识别
        if not data.camera_info.enable_action_recognition or self.count % self.prediction_interval != 0:
            self.count += 1
            self.count = self.count % self.prediction_interval
            data.action_info = deepcopy(self.action_info)
            return

        # 2. 识别
        self.count += 1
        self.predict_action(data)
        data.action_info = deepcopy(self.action_info)

        # 3. 调试
        if self.config['enable_debug']:
            logger.info("Action Recognition Time {}".format(time.time() - start_time))
