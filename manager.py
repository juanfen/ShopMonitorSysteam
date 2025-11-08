import os
import time
from process import monitor
from multiprocessing import Process

from utils import full_init_insightface, incremental_init_insightface, ping_ip


class Manager:
    def __init__(self, config, info_queue):
        self.config = config
        self.info_queue = info_queue

        self.last_check = time.time()
        self.group_num = len(config['camera']['group'])
        self.monitor_list = []
        for i in range(self.group_num):
            process = Process(target=monitor, args=(config, i, info_queue))
            self.monitor_list.append(process)

        self.data_dir = os.path.join(
            config['static_root'],
            config['static']['face_images']
        )
        self.feat_dir = os.path.join(
            config['static_root'],
            config['static']['face_features']
        )
        # if not os.path.exists(self.feat_dir):
        #     full_init_insightface(self.data_dir, self.feat_dir, name=config['face recognition']['model'])

    def start(self):
        for process in self.monitor_list:
            process.start()

    def restart(self):
        # flag = True
        # if time.time() - self.last_check > self.config['manager']['restart_time']:
        #     flag = incremental_init_insightface(self.data_dir, self.feat_dir, name=self.config['face recognition']['model'])
        #     self.last_check = time.time()
        for i in range(self.group_num):
            # if not self.monitor_list[i].is_alive() or not flag:
            if not self.monitor_list[i].is_alive():
                self.monitor_list[i].terminate()
                self.monitor_list[i].join()
                self.monitor_list[i].close()
                process = Process(target=monitor, args=(self.config, i, self.info_queue))
                process.start()
                self.monitor_list[i] = process

    def stop(self):
        for process in self.monitor_list:
            process.terminate()
            process.join()
            process.close()
