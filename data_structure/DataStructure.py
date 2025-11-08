from typing import List
from dataclasses import dataclass, field


@dataclass
class CameraInfo:
    k = 0  # 行人跟踪的穿越线 x轴坐标

    rate = 1  # 摄像头采样率
    name = ""  # 摄像头名称
    port = ""  # 摄像头完整端口号

    device = "cuda:0"  # 是否使用cuda加速 需要tensorrt支持
    enable_reid = False  # 启用充实别 需要同组内多摄像头

    enable_writer = False  # 是否记录异常情况
    enable_crossed_writer = False  # 异常中的穿过跨越线是否需要记录
    enable_abnormal_writer = False  # 异常中的行为异常是否需要记录 待后期拓展
    cross_direction: List[str] = field(default_factory=list)  # 穿越的哪个方向会被记录 Left Right 仅支持这两个方向

    enable_face_record = False  # 是否启用人脸记录
    enable_face_recognition = False  # 是否启用人脸识别
    enable_silent_detected = False  # 无行人时系统是否静默
    enable_action_recognition = True  # 是否启用动作识别


@dataclass
class PersonInfo:
    person_id = 0  # 行人被分配的id
    person_roi = None  # 完整图像中被裁切出的指定id行人区域
    xyxy = ()  # 行人在图像上的坐标
    track: List[any] = field(default_factory=list)  # 跟踪记录
    color = (255, 0, 0)  # 当前行人分配的颜色
    movement = "None"  # 行人穿越方向
    crossed = False  # 该行人是否跨过穿越线
    create_time = None  # 创建的时间


@dataclass
class ReidPersonInfo:
    person_reid = -1  # 未reid准备的值
    embed = None  # 用于充实别的外观特征向量


@dataclass
class FacePersonInfo:
    person_name = "Detecting..."  # 行人姓名
    retry_times = 0  # 识别重试次数
    is_verified = False  # 行人身份信息是否已经被确认
    is_new = False  # 是否是新的需要被记录的脸
    image_path = ""  # 新录入的人需要保存一张人脸图像 仅在 is_new 为 True 时有值


# @dataclass
# class ActionPersonInfo:
#     action_label = "None"  # drinking, eating, hiding, tearing
#     action_confidence = 0.0  # 置信度
#     action_roi = None  # 用于动作识别的ROI区域
#     is_abnormal = False  # 是否异常行为（根据业务逻辑）


@dataclass
class TrackerInfo:
    person_list: List[PersonInfo] = field(default_factory=list)
    person_lock = False  # 行人锁
    drop_ids: List[int] = field(default_factory=list)  # 跟踪弃用的行人id列表
    is_gathering = False  # 行人是否聚众


@dataclass
class ReidInfo:
    person_list: List[ReidPersonInfo] = field(default_factory=list)
    reid_drop_ids: List[int] = field(default_factory=list)  # reid后跟踪弃用的行人id列表


@dataclass
class FaceInfo:
    person_list: List[FacePersonInfo] = field(default_factory=list)


@dataclass
class WriterInfo:
    image_name = ""
    video_name = ""


@dataclass
class ActionInfo:
    # person_list: List[ActionPersonInfo] = field(default_factory=list)
    # abnormal_count = 0  # 异常行为计数
    action_label = "None"  # drinking, eating, hiding, tearing
    action_confidence = 0.0  # 置信度
    # action_roi = None  # 用于动作识别的ROI区域
    is_abnormal = False  # 是否异常行为（根据业务逻辑）


@dataclass
class DataStructure:
    def __init__(self):
        self.frame = None
        self.clip = None
        self.camera_info = None
        self.tracker_info = TrackerInfo()
        self.reid_info = ReidInfo()
        self.face_info = FaceInfo()
        self.writer_info = WriterInfo()
        self.action_info = None

    def get_tracker_memory_data(self):
        person_list = []
        for p, rp in zip(self.tracker_info.person_list, self.reid_info.person_list):
            item = {
                "person_id": rp.person_reid,
                "movement": p.movement,
                "crossed": p.crossed,
                "create_time": p.create_time,
            }
            person_list.append(item)
        return person_list

    def get_face_memory_data(self):
        person_list = []
        for rp, fp in zip(self.reid_info.person_list, self.face_info.person_list):
            item = {
                "person_id": rp.person_reid,
                "person_name": fp.person_name,
                "is_verified": fp.is_verified,
                "is_new": fp.is_new,
                "image_path": fp.image_path
            }
            person_list.append(item)
        return person_list

    # 添加获取动作记忆数据的方法
    def get_action_memory_data(self):
        person_list = []
        for rp, ap in zip(self.reid_info.person_list, self.action_info.person_list):
            item = {
                "person_id": rp.person_reid,
                "action_label": ap.action_label,
                "action_confidence": ap.action_confidence,
                "is_abnormal": ap.is_abnormal
            }
            person_list.append(item)
        return person_list
