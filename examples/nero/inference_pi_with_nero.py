import logging
import time
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from utils import FpsCounter
from openpi_client import image_tools
from recorder import Recorder 
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs

# Import the actual client we just copied from the teleop folder
from nero_interface_client import NeroDualArmClient

logging.basicConfig(level=logging.INFO, format="%(message)s")

home = Path.home()

def rotvec_to_rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotation_vector).as_matrix()

def rotation_matrix_to_rotvec(rot_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_matrix).as_rotvec()

def apply_delta_rotation(current_rotvec: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Apply delta rotation to current rotation using rotation matrices."""
    current_rot = rotvec_to_rotation_matrix(current_rotvec)
    delta_rot = rotvec_to_rotation_matrix(delta_rotvec)
    new_rot = delta_rot @ current_rot
    return rotation_matrix_to_rotvec(new_rot)

def update_latest_symlink(target: Path, link_name: Path):
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)

class Inference:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model = cfg["model"]
        self.model_config = _config.get_config(model["name"])
        chk_dir = str(model["checkpoint_dir"])
        self.checkpoint_dir = Path(chk_dir) if chk_dir.startswith("/") else home / chk_dir
        
        # Camera config (3 cameras for complete Nero vision setup)
        cam = cfg.get("cameras", {})
        self.left_wrist_cam_serial = str(cam.get("left_wrist_cam_serial", ""))
        self.right_wrist_cam_serial = str(cam.get("right_wrist_cam_serial", ""))
        self.exterior_cam_serial = str(cam.get("exterior_cam_serial", ""))
        self.cam_fps = cam.get("fps", 30)

        video = cfg.get("video", {"fps": 15, "visualize": True})
        self.video_fps = video.get("fps", 15)
        self.visualize = video.get("visualize", True)

        robot = cfg.get("robot", {})
        self.robot_ip = robot.get("ip", "127.0.0.1")
        self.robot_port = robot.get("port", 4242)
        self.initial_left_joints = np.asarray(robot.get("initial_left_joints", np.zeros(7)), dtype=np.float32)
        self.initial_right_joints = np.asarray(robot.get("initial_right_joints", np.zeros(7)), dtype=np.float32)
        self.dry_run = cfg.get("dry_run", False)
        
        run_cfg = cfg.get("run", {})
        self.action_fps = run_cfg.get("action_fps", robot.get("action_fps", 20))
        self.action_horizon = run_cfg.get("action_horizon", robot.get("action_horizon", 10))

        gripper = cfg.get("gripper", {})
        self.close_threshold = gripper.get("close_threshold", 0.05)
        self.gripper_force = gripper.get("gripper_force", 10.0)
        self.gripper_speed = gripper.get("gripper_speed", 0.1)
        self.gripper_reverse = gripper.get("gripper_reverse", False)

        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "delta_ee")
        self.ee_action_scale = action_mode.get("ee_action_scale", 1.0)

        task = cfg.get("task", {"description": run_cfg.get("task_description", "pick and place")})
        self.task_description = task.get("description", "pick and place")
        
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')

        base_dir = Path(__file__).parent
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        left_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_left_wrist_{time_str}.mp4"
        right_wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_right_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"

        self.recorder = Recorder(log_path=log_path, video_path=[left_wrist_video, right_wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)
        
        update_latest_symlink(log_path, latest_path)
        self.fps_action = FpsCounter(name="action")
        self.robot_client = None
        self.cameras = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to Nero dual-arm robot."""
        if self.dry_run:
            logging.info("[DUMMY] Dry run: Skipping robot connection.")
            return

        try:
            logging.info(f"\n===== [ROBOT] Connecting to Nero dual-arm robot at {self.robot_ip}:{self.robot_port} =====")
            self.robot_client = NeroDualArmClient(ip=self.robot_ip, port=self.robot_port)
            if self.robot_client.server is None:
                raise ConnectionError("Server connection failed.")

            left_pose = self.robot_client.left_robot_get_ee_pose()
            right_pose = self.robot_client.right_robot_get_ee_pose()
            lgrip = self.robot_client.left_gripper_get_state().get("width", 0.0)
            rgrip = self.robot_client.right_gripper_get_state().get("width", 0.0)

            logging.info(f"[STATE] Left Arm Pose: {left_pose[:3]} | R: {left_pose[3:]}")
            logging.info(f"[STATE] Right Arm Pose: {right_pose[:3]} | R: {right_pose[3:]}")
            logging.info(f"[STATE] Left Gripper width: {lgrip} | Right Gripper width: {rgrip}")
            logging.info("===== [ROBOT] Nero initialized successfully =====\n")
        except Exception as e:
            logging.error(f"===== [ERROR] Failed to connect to Nero robot: {e} =====")

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        if self.dry_run:
            logging.info("[DUMMY] Dry run: Skipping camera connection.")
            return

        try:
            logging.info("\n===== [CAMERAS] Connecting to Realsense Cameras =====")
            configs = {}
            if self.exterior_cam_serial:
                configs["exterior_image"] = RealSenseCameraConfig(camera_index=self.exterior_cam_serial, fps=self.cam_fps, color_mode=ColorMode.RGB)
            if self.left_wrist_cam_serial:
                configs["left_wrist_image"] = RealSenseCameraConfig(camera_index=self.left_wrist_cam_serial, fps=self.cam_fps, color_mode=ColorMode.RGB)
            if self.right_wrist_cam_serial:
                configs["right_wrist_image"] = RealSenseCameraConfig(camera_index=self.right_wrist_cam_serial, fps=self.cam_fps, color_mode=ColorMode.RGB)

            if configs:
                self.cameras = make_cameras_from_configs(configs)
                self.cameras.connect()
                logging.info(f"[CAMERAS] Connected: {list(configs.keys())}")
            else:
                logging.error("[CAMERAS] No cameras configured in config file")
        except Exception as e:
            logging.error(f"===== [ERROR] Failed to connect to cameras: {e} =====")

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # -> 14D
        state = np.concatenate((
            np.asarray(obs["left_ee_pose"], dtype=np.float32),
            np.asarray(obs["right_ee_pose"], dtype=np.float32),
            np.asarray([obs["left_gripper_position"]], dtype=np.float32),
            np.asarray([obs["right_gripper_position"]], dtype=np.float32),
        ))

        ext = image_tools.resize_with_pad(obs.get("exterior_image", np.zeros((480, 640, 3), dtype=np.uint8)), 224, 224)
        lw = image_tools.resize_with_pad(obs.get("left_wrist_image", np.zeros((480, 640, 3), dtype=np.uint8)), 224, 224)
        rw = image_tools.resize_with_pad(obs.get("right_wrist_image", np.zeros((480, 640, 3), dtype=np.uint8)), 224, 224)

        return {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(ext),
            "observation/wrist_image": image_tools.convert_to_uint8(lw),
            "observation/right_wrist_image": image_tools.convert_to_uint8(rw),
            "prompt": obs.get("prompt", ""),
        }

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        obs = {}

        if self.robot_client:
            obs["left_ee_pose"] = self.robot_client.left_robot_get_ee_pose()
            obs["right_ee_pose"] = self.robot_client.right_robot_get_ee_pose()
            obs["left_gripper_position"] = self.robot_client.left_gripper_get_state().get("width", 0.0)
            obs["right_gripper_position"] = self.robot_client.right_gripper_get_state().get("width", 0.0)
        else:
            obs["left_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["right_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["left_gripper_position"] = 0.0
            obs["right_gripper_position"] = 0.0

        if self.cameras:
            frames = self.cameras.read()
            for k in ["exterior_image", "left_wrist_image", "right_wrist_image"]:
                if k in frames:
                    obs[k] = frames[k]
        else:
            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            obs["exterior_image"] = img
            obs["left_wrist_image"] = img
            obs["right_wrist_image"] = img

        obs["prompt"] = self.task_description

        return self._transfer_obs_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray):
        if self.dry_run:
            logging.info("[DUMMY] Executed 14D Action Block.")
            return

        if self.robot_client is None:
            return

        if self.action_mode == "delta_ee":
            self._execute_delta_ee_actions(actions)
        else:
            logging.error(f"[ERROR] Unsupported action mode {self.action_mode} for NERO.")

    def _execute_delta_ee_actions(self, actions: np.ndarray):
        """Execute delta end-effector actions on dual arm.
        Action format (14D): [l_dx, l_dy, l_dz, l_drx, l_dry, l_drz,
                              r_dx, r_dy, r_dz, r_drx, r_dry, r_drz,
                              l_grip, r_grip]"""
        cur_left_ee = self.robot_client.left_robot_get_ee_pose()
        cur_right_ee = self.robot_client.right_robot_get_ee_pose()
        
        cur_l_pos, cur_l_rot = cur_left_ee[:3].copy(), cur_left_ee[3:6].copy()
        cur_r_pos, cur_r_rot = cur_right_ee[:3].copy(), cur_right_ee[3:6].copy()

        for action in actions[:self.action_horizon]:
            start_time = time.perf_counter()

            # Left Arm Deltas
            d_l_pos, d_l_rot = action[0:3] * self.ee_action_scale, action[3:6] * self.ee_action_scale
            tgt_l_pose = np.concatenate([cur_l_pos + d_l_pos, apply_delta_rotation(cur_l_rot, d_l_rot)])

            # Right Arm Deltas
            d_r_pos, d_r_rot = action[6:9] * self.ee_action_scale, action[9:12] * self.ee_action_scale
            tgt_r_pose = np.concatenate([cur_r_pos + d_r_pos, apply_delta_rotation(cur_r_rot, d_r_rot)])

            self.robot_client.servo_p("left_robot", tgt_l_pose, delta=False)
            self.robot_client.servo_p("right_robot", tgt_r_pose, delta=False)

            cur_left_ee = self.robot_client.left_robot_get_ee_pose()
            cur_right_ee = self.robot_client.right_robot_get_ee_pose()
            cur_l_pos, cur_l_rot = cur_left_ee[:3].copy(), cur_left_ee[3:6].copy()
            cur_r_pos, cur_r_rot = cur_right_ee[:3].copy(), cur_right_ee[3:6].copy()

            # Control grippers
            l_grip_cmd = 0 if action[12] < self.close_threshold else 1
            r_grip_cmd = 0 if action[13] < self.close_threshold else 1
            if self.gripper_reverse:
                l_grip_cmd, r_grip_cmd = 1 - l_grip_cmd, 1 - r_grip_cmd

            self.robot_client.left_gripper_goto(width=l_grip_cmd*0.0801, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=r_grip_cmd*0.0801, force=self.gripper_force)

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()

        if self.robot_client:
            if np.any(self.initial_left_joints):
                self.robot_client.left_robot_move_to_joint_positions(self.initial_left_joints)
            if np.any(self.initial_right_joints):
                self.robot_client.right_robot_move_to_joint_positions(self.initial_right_joints)

            self.robot_client.left_gripper_goto(width=0.0801, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=0.0801, force=self.gripper_force)
        
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation mapped keys: {obs.keys()}")
        
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        logging.info("Warming up the model...")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                t0 = time.perf_counter()
                obs = self.get_obs_state()
                result = policy.infer(obs)
                self.execute_actions(result["actions"])
                self.recorder.submit_actions(result["actions"], infer_time, obs.get("prompt", ""))
                self.recorder.submit_obs(obs)
                
                logging.info(f"[STATE] Loop rate: {1 / (time.perf_counter() - t0):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detected. Stopping.")
        except Exception as e:
            logging.error(f"[ERROR] Loop error: {e}")
            raise e
        finally:
            if self.robot_client:
                self.robot_client.close()
            if self.cameras:
                self.cameras.disconnect()

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_nero_pi.yaml"
    inference = Inference(config_path)
    inference.run()

if __name__ == "__main__":
    main()
