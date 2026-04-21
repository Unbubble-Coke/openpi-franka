# recorder.py
import threading
import queue
import yaml
import time
import cv2
import av
import numpy as np
from pathlib import Path

class Recorder:
    """This class handles asynchronous logging and visualization for Dual-Arm Nero."""

    def __init__(self, log_path: Path, video_path: list, display_fps: int = 15, visualize: bool = False):
        self.visualize = visualize
        self.log_path = log_path

        # video paths: left_wrist, right_wrist, exterior
        self.left_wrist_video = video_path[0]
        self.right_wrist_video = video_path[1]
        self.exterior_video = video_path[2]

        self.display_fps = display_fps
        
        # safe queues
        self.queue_log = queue.Queue(maxsize=1)
        self.queue_vis = queue.Queue(maxsize=1)

        # store frames for video
        self.frames_ext = []
        self.frames_left_wrist = []
        self.frames_right_wrist = []

        # start threads
        threading.Thread(target=self._logger_thread, daemon=True).start()
        threading.Thread(target=self._visualizer_thread, daemon=True).start()

    def _logger_thread(self):
        while True:
            data = self.queue_log.get()  # blocking
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
                    f.write(' ' + '-' * 60 + '\n')
                    f.flush()
            except Exception as e:
                print(f"[Recorder Logger ERROR] {e}")

    def _visualizer_thread(self):
        last_frame_time = 0
        while True:
            try:
                obs = self.queue_vis.get()
                ext = self.to_bgr(obs["observation/image"])
                l_wrist = self.to_bgr(obs["observation/wrist_image"])
                r_wrist = self.to_bgr(obs.get("observation/right_wrist_image", np.zeros_like(l_wrist)))
                
                self.frames_ext.append(ext.astype(np.uint8))
                self.frames_left_wrist.append(l_wrist.astype(np.uint8))
                self.frames_right_wrist.append(r_wrist.astype(np.uint8))

                if self.visualize:
                    combined = np.hstack((l_wrist, ext, r_wrist))
                    cv2.imshow("Left Wrist | Exterior | Right Wrist", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.time() - last_frame_time
                if elapsed < 1 / self.display_fps:
                    time.sleep(1 / self.display_fps - elapsed)
                last_frame_time = time.time()
            except Exception as e:
                print(f"[VIS ERROR] {e}")
                time.sleep(0.1)
    
    def _encode(self, frames, out_path: Path, vcodec: str = "libx264", crf: int = 23, preset="medium"):
        h, w, _ = frames[0].shape
        container = av.open(str(out_path), "w")
        stream = container.add_stream(vcodec, rate=self.display_fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": str(preset)}

        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"{out_path.name}: {size_mb:.2f} MB")
        return size_mb
        
    def save_video(self):
        if not self.frames_ext or not self.frames_left_wrist or not self.frames_right_wrist:
            print("No frames to save.")
            return

        print("\nSaving exterior camera video...")
        self._encode(self.frames_ext, self.exterior_video, vcodec="libx264", crf=23, preset="veryslow")

        print("\nSaving left wrist camera video...")
        self._encode(self.frames_left_wrist, self.left_wrist_video, vcodec="libx264", crf=23, preset="veryslow")

        print("\nSaving right wrist camera video...")
        self._encode(self.frames_right_wrist, self.right_wrist_video, vcodec="libx264", crf=23, preset="veryslow")

    def to_bgr(self, img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def submit_actions(self, actions, infer_time: int, prompt: str = ""):
        try:
            actions_list = np.round(np.asarray(actions), 3).tolist()
        except Exception:
            actions_list = list(actions)
        
        try:
            self.queue_log.put_nowait({
                "infer_time": infer_time,
                "prompt": prompt,
                "actions": actions_list
            })
        except queue.Full:
            pass

    def submit_obs(self, obs):
        try:
            self.queue_vis.put_nowait(obs)
        except queue.Full:
            pass
