import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50


class SlowFast(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 加载预训练的SlowFast R50模型
        self.model = slowfast_r50(pretrained=pretrained)

        # 替换分类头以适应你的类别数
        in_features = self.model.blocks[6].proj.in_features
        self.model.blocks[6].proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
