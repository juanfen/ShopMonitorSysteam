import av
import cv2
from loguru import logger


def save_image(
        image_name,
        image
):
    try:
        cv2.imwrite(image_name, image)
        logger.info("图片写入完成\n图片路径 {}".format(image_name))
    except Exception as e:
        logger.info("图片写入失败\n错误信息 {}".format(e))


def video_composition(
        video_name,
        frames,
        fps=6,
        imgsize=(1920, 1080)
):
    if not frames:
        logger.warning("帧列表为空，未生成视频。")
        return

    logger.info("开始写入视频: {}".format(video_name))

    try:
        output = av.open(video_name, mode='w')
        stream = output.add_stream('libx264', rate=fps)  # 用 libx264 更兼容
        stream.width, stream.height = imgsize
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'medium'}  # 可调参数提升质量

        for i, f in enumerate(frames):
            # 校验帧尺寸是否一致，不一致则 resize
            if f.shape[1] != imgsize[0] or f.shape[0] != imgsize[1]:
                f = cv2.resize(f, imgsize)

            # 确保是 BGR 格式
            if len(f.shape) == 2:  # 灰度图转彩色
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            temp = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame = av.VideoFrame.from_ndarray(temp, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)

        # Flush 编码器
        for packet in stream.encode():
            output.mux(packet)

        logger.info("视频写入完成: {}".format(video_name))
    except Exception as e:
        logger.error("视频写入出错: {}".format(str(e)))
    finally:
        if 'output' in locals():
            output.close()
