import time
import tkinter as tk
import multiprocessing

from utils import check_config
from ui import MultiCameraApp
from manager import Manager


def main():
    config = check_config()
    info_queue = multiprocessing.Queue()
    manager = Manager(config, info_queue)
    manager.start()

    if config['enable_ui']:
        root = tk.Tk()
        MultiCameraApp(root, config, info_queue)
        root.mainloop()

    while True:
        manager.restart()
        time.sleep(0.1)
        # if keyboard.is_pressed('q'):
        #     logger.info("程序正在退出")
        #     break

    manager.stop()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
