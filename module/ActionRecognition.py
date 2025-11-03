import os
import time
import numpy as np

import torch
from torchvision.transforms import v2
from loguru import logger

from .net import SlowFast


class ActionRecognition:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_config = config.get('action recognition', {})

        # 模型参数
        self.model_path = os.path.join(config['static_root'], config['static']['model'], config['action recognition']['model_path'])
        self.num_classes = self.action_config['num_classes']
        self.threshold = self.action_config['threshold']
        self.clip_length = self.action_config['clip_length']
        self.input_size = self.action_config['input_size']
        self.prediction_interval = self.action_config['prediction_interval']

        # 推理准备
        self.count = 0
        self.model = None
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
        # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
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
        print(len(fast_data))
        fast_frames = self.transform(torch.tensor(fast_data))
        slow_frames = fast_frames[::4].clone()

        slow_path = slow_frames.permute(1, 0, 2, 3)
        fast_path = fast_frames.permute(1, 0, 2, 3)

        return slow_path, fast_path

    def predict_action(self, data):
        with torch.no_grad():
            # 1. 预处理帧
            slow_input, fast_input = self.preprocess_frames(data.clip)
            slow_input = slow_input.to(self.device)
            fast_input = fast_input.to(self.device)

            # 2. 模型预测
            outputs = self.model([slow_input, fast_input])
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            data.action_info.action_confidence = probabilities[0][predicted_class].item()
            all_probabilities = probabilities[0].cpu().numpy()

            # 3. 应用置信度阈值
            if data.action_info.action_confidence < self.threshold:
                data.action_info.action_label = "None"
                data.action_info.is_abnormal = False
            else:
                data.action_info.action_label = self.action_classes[predicted_class]
                data.action_info.is_abnormal = True

            # 4. 调试模式下输出详细预测信息
            if self.config['enable_debug']:
                # 构建概率详情字符串
                prob_details = []
                for i, prob in enumerate(all_probabilities):
                    prob_details.append(f"{self.action_classes[i]}: {prob:.4f}")
                prob_str = ", ".join(prob_details)

                logger.debug(f"动作识别预测结果: {data.action_info.action_label} (置信度: {data.action_info.action_confidence:.4f})")
                logger.debug(f"所有类别概率: [{prob_str}]")
                logger.debug(f"是否异常行为: {data.action_info.is_abnormal}")
                logger.debug(f"阈值: {self.threshold}, 预测类别索引: {predicted_class}")

    def get_action_result(self, data):
        start_time = time.time()

        # 1. 判断是否需要进行动作识别
        if not data.camera_info.enable_action_recognition or self.count % self.prediction_interval != 0:
            self.count += 1
            self.count  = self.count % self.prediction_interval
            # for _ in data.tracker_info.person_list:
            #     data.action_info.person_list.append(ActionPersonInfo())
            return

        # 2. 识别
        self.predict_action(data)

        # 3. 调试
        if self.config['enable_debug']:
            logger.info("Action Recognition Time {}".format(time.time() - start_time))
