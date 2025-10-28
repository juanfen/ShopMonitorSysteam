import os
import time

import av
import cv2
import sys
import yaml
import requests
import platform

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def ping_ip(ip):
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = f"ping {param} 1 {ip}"
    return os.system(command) == 0  # 返回0表示成功


def init_insight_face(feature_dir, name='buffalo_l'):
    app = FaceAnalysis(name=name, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    ctx_id = 0 if torch.cuda.is_available() else -1
    app.prepare(ctx_id, det_size=(640, 640))  # ctx_id=-1 用 CPU 0 用 GPU

    face_db = {}
    logger.info(f"📂 开始加载人脸特征库：{feature_dir}")
    files = [f for f in os.listdir(feature_dir) if f.endswith('_feat.npy')]
    for i, filename in enumerate(tqdm(files)):
        name = filename.replace('_feat.npy', '')
        path = os.path.join(feature_dir, filename)
        try:
            feature = np.load(path)
            if feature.shape != (512,):
                logger.error(f"❌ 无效特征维度{feature.shape} 跳过：{filename}")
                continue
            face_db[name] = feature
        except Exception as e:
            logger.error(f"[{i + 1}/{len(files)}] ❌ 加载失败: {filename} 原因: {e}")
    logger.info(f"✅ 完成人脸特征加载，共加载 {len(face_db)} 人")
    return app, face_db


def full_init_insightface(image_dir, output_dir, name='buffalo_l'):
    os.makedirs(output_dir, exist_ok=True)

    # 初始化人脸识别模型
    app = FaceAnalysis(name=name)
    ctx_id = 0 if torch.cuda.is_available() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))  # 使用 GPU (ctx_id=0)，CPU 则设置为 -1

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    error_files = []
    error = []
    for path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(path)

        if img is None:
            error.append(path.split('\\')[-1].split('/')[-1])
            error_files.append(f"⚠ 无法读取图像: {path}")
            continue

        faces = app.get(img)
        if not faces:
            error.append(path.split('\\')[-1].split('/')[-1])
            error_files.append(f"❌ 未检测到人脸: {path}")
            continue

        feature = faces[0].normed_embedding
        filename = path.split('/')[-1].split('\\')[-1].split('.')[0]
        feat_path = os.path.join(output_dir, f"{filename}_feat.npy")
        np.save(feat_path, feature)

    if error_files:
        for it in error_files:
            logger.error(it)
        print(error)
    else:
        logger.info("✅ 所有人脸特征提取完成！")


def incremental_init_insightface(image_dir, output_dir, name='buffalo_l'):
    img_num = len(os.listdir(image_dir))
    feat_num = len(os.listdir(output_dir))
    flag = True if img_num == feat_num else False
    if not flag:
        full_init_insightface(image_dir, output_dir, name)
    return flag


def check_movement(points, k):
    # 判断整体方向
    direction = "None"
    crossed = False
    if len(points) > 1:
        direction = "Left" if points[-1][0] - points[0][0] < 0 else "Right"

        # 检测是否穿越x=k
        for i in range(len(points) - 1):
            x1, x2 = points[i][0], points[i + 1][0]
            if (x1 - k) * (x2 - k) <= 0:
                crossed = True
                break

    return direction, crossed


def export_pt_to_engine(path):
    model = YOLO(path)
    model.export(format="engine", dynamic=True, half=True)


def set_logger(name, format):
    logger.remove()
    logger.add(sink=sys.stdout, format=format, level="DEBUG")

    logger.add(sink=name, format=format, rotation="500 MB", retention="30 days", enqueue=True, encoding="utf-8")


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("创建文件夹成功 {}".format(path))
            return
        except OSError("创建文件夹失败 {}".format(path)):
            raise
    print("文件夹已存在 {}".format(path))


def check_pt_or_engine(fp, files, file):
    if file in files:
        return
    elif file.endswith(".pt"):
        raise FileNotFoundError("检测模型缺失 {}".format(file))
    elif file.endswith(".engine"):
        if not torch.cuda.is_available():
            raise RuntimeError("该设备没有GPU或PyTorch不支持GPU")
        files_drop_end = [item.split('.')[0] for item in files]
        file_drop_end = file.split('.')[0]
        if file_drop_end not in files_drop_end:
            raise FileNotFoundError("检测模型缺失 {}".format(file))
        try:
            export_pt_to_engine(os.path.join(fp, file_drop_end + '.pt'))
        except RuntimeError("从pt文件导出engine失败"):
            raise


def check_config():
    # 读取config.yaml
    print("正在加载配置文件...")
    if not os.path.exists("config.yaml"):
        raise FileNotFoundError("配置文件不存在 请检查路径或config文件夹下是否存在config.yaml")
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    print("成功加载配置文件...\n")

    # 检测静态文件
    print("正在检查配置文件...")
    root = config['static_root']
    print("正在检查{}文件夹".format(root))
    if not os.path.exists(config['static_root']):
        raise FileNotFoundError("static文件夹不存在 请检查路径或static文件夹是否存在")

    # 检查 log 保存路径
    log_path = os.path.join(root, config['static']['log'])
    print("正在检测{}文件夹".format(log_path))
    create_dir(log_path)
    print("{}文件夹检测完成".format(log_path))

    # 检查 图片 保存路径
    image_path = os.path.join(root, config['static']['image'])
    print("正在检测{}文件夹".format(image_path))
    create_dir(image_path)
    print("{}文件夹检测完成".format(image_path))

    # 检查 视频 保存路径
    video_path = os.path.join(root, config['static']['video'])
    print("正在检测{}文件夹".format(video_path))
    create_dir(video_path)
    print("{}文件夹检测完成".format(video_path))

    model_path = os.path.join(root, config['static']['model'])
    print("正在检测{}文件夹".format(model_path))
    pt_file = os.listdir(model_path)
    check_pt_or_engine(model_path, pt_file, config['people tracker']['model_path'])
    print("{}文件夹检测完成".format(model_path))

    print("{}文件夹检测完成".format(root))
    print("初始化完成")
    return config


def check_gathering_by_distance(centroids, distance_threshold=50, ratio_threshold=0.6):
    """
    基于最近邻距离判断聚众行为
    参数:
        centroids: 所有目标中心点的NumPy数组，形状为(N, 2)，N为目标数量。
        distance_threshold: 判定为"过近"的距离阈值（像素单位）。
        ratio_threshold: 判定发生聚集时，过近目标数量占总目标数量的最小比例。
    返回:
        is_gathering: 布尔值，是否存在聚众行为。
        detailed_info: 包含详细信息的字典，如最近邻距离数组、过近目标的比例等。
    """
    if len(centroids) < 2:
        return False, {"reason": "目标数量少于2，无法计算最近邻距离"}

    # 计算每个点到其最近邻点的距离
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(centroids)
    distances, _ = nbrs.kneighbors(centroids)
    # 每个点与最近邻点的距离是distances中的第二列（第一列是自身，距离为0）
    min_distances = distances[:, 1]

    # 统计最近邻距离小于阈点的点的比例
    close_count = np.sum(min_distances < distance_threshold)
    close_ratio = close_count / len(centroids)

    # 判断是否发生聚集
    is_gathering = close_ratio > ratio_threshold

    detailed_info = {
        "min_distances": min_distances,
        "close_count": close_count,
        "close_ratio": close_ratio,
        "distance_threshold": distance_threshold,
        "ratio_threshold": ratio_threshold
    }

    return is_gathering, detailed_info


def check_gathering_by_clustering(centroids, eps=50, min_samples=3, cluster_size_threshold=5):
    """
    使用DBSCAN聚类算法判断聚众行为
    参数:
        centroids: 所有目标中心点的NumPy数组，形状为(N, 2)。
        eps: 邻域半径，同一簇中两个样本之间的最大距离。
        min_samples: 形成一个核心点所需的最小样本数。
        cluster_size_threshold: 判定为“聚众”的簇所需的最小目标数。
    返回:
        is_gathering: 布尔值，是否存在聚众行为。
        detailed_info: 包含详细信息的字典，如聚类标签、各簇大小等。
    """
    s = time.time()
    if len(centroids) < min_samples:
        return False, {"reason": f"目标数量少于最小样本数 {min_samples}，无法形成有效聚类"}

    # 执行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_

    # 统计每个簇的大小（忽略噪声点，标签为-1）
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # 判断是否存在超过阈值的簇
    large_clusters = [label for label, size in cluster_sizes.items() if size >= cluster_size_threshold]
    is_gathering = len(large_clusters) > 0

    detailed_info = {
        "cluster_labels": labels,
        "cluster_sizes": cluster_sizes,
        "large_clusters": large_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "cluster_size_threshold": cluster_size_threshold
    }
    e = time.time()
    # print(e - s)
    processing_time = e - s
    logger.debug(f"DBSCAN聚类耗时: {processing_time:.3f}s")
    return is_gathering, detailed_info


if __name__ == '__main__':
    test_data = np.array(
        [
            (30, 30),
            (40, 40),
            (30, 40),
            (30, 20),
            (20, 30),
            (120, 90),
            (200, 70),
        ]
    )
    f = check_gathering_by_clustering(test_data)
    print(f)
