import cv2
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
import threading
import time
import queue

class MultiCameraApp:
    def __init__(self, root, config, frame_queue, image_size=(640, 480)):
        self.root = root
        self.camera_ips = []
        for group in config['camera']['group']:
            for key in group.keys():
                for i in range(len(group[key]['ip'])):
                    self.camera_ips.append(group[key]['ip'][i]['name'])
        self.frame_queue = frame_queue
        self.image_size = image_size
        self.frames = {}
        self.labels = {}
        self.running = True

        self.root.title("我是py大跌")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create a grid layout to hold camera feeds
        rows = len(self.camera_ips) // 2 + len(self.camera_ips) % 2
        for i, ip in enumerate(self.camera_ips):
            row, col = divmod(i, 2)

            frame = Frame(root)
            frame.grid(row=row, column=col, padx=5, pady=5)

            label = Label(frame)
            label.pack()
            self.labels[ip] = label

            ip_label = Label(frame, text=ip, bg="black", fg="white", font=("Arial", 10))
            ip_label.pack()

        self.update_frames()

    def update_frames(self):
        if not self.running:
            return

        try:
            frames_processed = 0
            max_frames_per_cycle = 5  # Limit the number of frames processed in each cycle

            while not self.frame_queue.empty() and frames_processed < max_frames_per_cycle:
                frame_data = self.frame_queue.get(timeout=0.1)
                ip = frame_data.get("camera_ip")
                frame = frame_data.get("frame")

                if ip in self.labels and frame is not None:
                    # Resize the frame to the specified image size
                    frame = cv2.resize(frame, self.image_size)
                    # Convert the frame to RGB for Tkinter display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.labels[ip].imgtk = imgtk
                    self.labels[ip].configure(image=imgtk)

                frames_processed += 1
        except queue.Empty:
            pass  # No frames to process within the timeout period
        except Exception as e:
            print(f"Error processing frames: {e}")

        self.root.after(2, self.update_frames)

    def stop(self):
        self.running = False
        self.root.destroy()

    def on_closing(self):
        self.stop()

if __name__ == "__main__":
    # Example camera IPs and frame queue
    camera_ips = ["192.168.0.101", "192.168.0.102"]
    frame_queue = queue.Queue()

    # Example of how frames could be added to the queue (this would normally come from another thread)
    def mock_frame_producer(queue):
        import cv2
        import numpy as np
        while True:
            for ip in camera_ips:
                fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                queue.put({"camera_ip": ip, "frame": fake_frame})
            time.sleep(0.05)

    threading.Thread(target=mock_frame_producer, args=(frame_queue,), daemon=True).start()

    root = tk.Tk()
    app = MultiCameraApp(root, camera_ips, frame_queue, image_size=(640, 480))
    root.mainloop()
