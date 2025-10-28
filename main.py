import time
import tkinter as tk
import multiprocessing

from utils import check_config
from ui import MultiCameraApp
from manager import Manager

import warnings
# 忽略 InsightFace 和 NumPy 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# 2. 主函数执行流程
def main():
    config = check_config()                 # 检查配置文件
    info_queue = multiprocessing.Queue()    # 创建进程间通信队列
    manager = Manager(config, info_queue)   # 创建管理器进程
    manager.start()                         # 启动管理器

    if config['enable_ui']:
        root = tk.Tk()                      # 创建主窗口
        MultiCameraApp(root, config, info_queue)  # 创建多摄像头应用界面
        root.mainloop()                      # 启动GUI事件循环

    while True:
        manager.restart()                    # 重启管理器进程
        time.sleep(0.1)                       # 短暂延迟
        # if keyboard.is_pressed('q'):
        #     logger.info("程序正在退出")
        #     break

    manager.stop()


if __name__ == "__main__":
    # 1. 设置多进程启动方式
    multiprocessing.set_start_method("spawn", force=True) # 确保进程安全
    main()
