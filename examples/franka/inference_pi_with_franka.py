import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import yaml
import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R
from utils import FpsCounter
from openpi_client import image_tools
from recorder import Recorder 
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
from franka_interface_client import FrankaInterfaceClient

home = Path.home()

def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """Convert Euler angles (rx, ry, rz) to rotation matrix.
    
    Uses 'XYZ' convention (intrinsic rotations about X, then Y, then Z).
    This matches the Franka robot's convention.
    
    Args:
        euler_angles: Euler angles in radians [rx, ry, rz]
    
    Returns:
        3x3 rotation matrix
    """
    return R.from_euler('XYZ', euler_angles).as_matrix()

def rotation_matrix_to_euler(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles.
    
    Uses 'XYZ' convention (intrinsic rotations about X, then Y, then Z).
    
    Args:
        rot_matrix: 3x3 rotation matrix
    
    Returns:
        Euler angles in radians [rx, ry, rz]
    """
    return R.from_matrix(rot_matrix).as_euler('XYZ')

def apply_delta_rotation(current_euler: np.ndarray, delta_euler: np.ndarray) -> np.ndarray:
    """Apply delta rotation to current rotation using rotation matrices.
    
    For delta EE actions, the delta rotation is defined in the 
    current end-effector frame (local frame), so we use:
        R_new = R_current @ R_delta
    
    Args:
        current_euler: Current Euler angles [rx, ry, rz] in radians
        delta_euler: Delta Euler angles [drx, dry, drz] in radians
    
    Returns:
        New Euler angles after applying delta rotation
    """
    # Convert current rotation to matrix
    current_rot = euler_to_rotation_matrix(current_euler)
    
    # Convert delta rotation to matrix (small rotation in local frame)
    delta_rot = euler_to_rotation_matrix(delta_euler)
    
    # Apply delta rotation in local frame: R_new = R_current @ R_delta
    new_rot = delta_rot @ current_rot
    
    # Convert back to Euler angles
    new_euler = rotation_matrix_to_euler(new_rot)
    
    return new_euler

def update_latest_symlink(target: Path, link_name: Path):
    """
    这个函数在机器人推理系统中用于维护一个"最新日志"的快捷方式，
    方便用户快速访问当前会话的日志信息。
    """
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)

class Inference:
    def __init__(self, config_path: Path):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Model config
        model = cfg["model"]
        self.model_config = _config.get_config(model["name"])
        self.checkpoint_dir = home / model["checkpoint_dir"]
        
        # Camera config
        cam = cfg["cameras"]
        self.wrist_cam_serial = cam["wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]

        # Robot config
        robot = cfg["robot"]
        self.robot_ip = robot["ip"]
        self.robot_port = robot["port"]
        self.initial_pose = np.asarray(robot["initial_pose"], dtype=np.float32)
        self.action_fps = robot["action_fps"]
        self.action_horizon = robot["action_horizon"]

        # Gripper config
        gripper = cfg["gripper"]
        self.close_threshold = gripper["close_threshold"]
        self.gripper_force = gripper["gripper_force"]
        self.gripper_speed = gripper["gripper_speed"]
        self.gripper_reverse = gripper["gripper_reverse"]

        # Action mode config
        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "joint")  # "joint" or "delta_ee"
        self.ee_action_scale = action_mode.get("ee_action_scale", 0.1)  # scale factor for delta ee actions

        # Task config
        task = cfg["task"]
        self.task_description = task["description"]
        
        # time stamps
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')

        # base dir
        base_dir = Path(__file__).parent
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        # create dir
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # log paths
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        # video paths
        wrist_video = video_dir / f"{self.task_description.replace(' ', '_')}_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"{self.task_description.replace(' ', '_')}_exterior_{time_str}.mp4"

        # Recorder  
        self.recorder = Recorder(log_path=log_path, video_path=[wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)
        
        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.robot_client = None
        self.cameras = None


    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to Franka robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Franka robot =====")
            self.robot_client = FrankaInterfaceClient(ip=self.robot_ip, port=self.robot_port)
            self.robot_client.gripper_initialize()

            # Joint positions
            joints = self.robot_client.robot_get_joint_positions().tolist()
            if joints and len(joints) == 7:
                formatted = [round(j, 4) for j in joints]
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                logging.info("[ERROR] Failed to read joint positions.")

            # TCP pose
            tcp_pose = self.robot_client.robot_get_ee_pose().tolist()
            if tcp_pose and len(tcp_pose) == 6:
                formatted_pose = [round(p, 4) for p in tcp_pose]
                logging.info(f"[ROBOT] Current TCP pose: {formatted_pose}")
                logging.info(
                    f"[ROBOT] Translation (m): x={formatted_pose[0]}, y={formatted_pose[1]}, z={formatted_pose[2]}"
                )
                logging.info(
                    f"[ROBOT] Rotation (rad): rx={formatted_pose[3]}, ry={formatted_pose[4]}, rz={formatted_pose[5]}"
                )
                logging.info("===== [ROBOT] Franka initialized successfully =====\n")
            else:
                logging.info("[ERROR] Failed to read TCP pose.")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to Franka robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)
    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")

            wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.wrist_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.exterior_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            camera_config = {"wrist_image": wrist_cfg, "exterior_image": exterior_cfg}
            self.cameras = make_cameras_from_configs(camera_config)

            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")

            logging.info("===== [CAM] Cameras initialized successfully =====\n")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to Franka policy format."""

        state = np.concatenate((
            np.asarray(obs["joint_positions"], dtype=np.float32),
            np.asarray([obs["gripper_position"]], dtype=np.float32),
        ))

        franka_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return franka_obs
    
    def _transfer_obs_delta_ee_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to Franka policy format."""

        state = np.concatenate((
            np.asarray(obs["ee_pose"], dtype=np.float32),
            np.asarray([obs["gripper_position"]], dtype=np.float32),
        ))

        franka_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return franka_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current observation from robot."""
        obs = {}

        # Robot state
        if self.robot_client:
            obs["joint_positions"] = self.robot_client.robot_get_joint_positions()
            obs["ee_pose"] = self.robot_client.robot_get_ee_pose()

        # Camera images
        if self.cameras:
            for name, cam in self.cameras.items():
                frame = cam.read()
                obs[name] = frame

        # Task description    
        if self.task_description:
            obs["prompt"] = self.task_description

        # Gripper state
        if self.robot_client:
            gripper_width = self.robot_client.gripper_get_state()["width"]
            gripper_state = max(0.0, min(1.0, gripper_width/0.0801))
            gripper_position = 0.0 if gripper_state  < self.close_threshold else 1.0
            obs["gripper_position"] = gripper_position
        
        # return self._transfer_obs_state(obs) 
        return self._transfer_obs_delta_ee_state(obs)        

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute the inferenced actions from the model."""
        if self.robot_client is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return
        
        if self.action_mode == "delta_ee":
            self._execute_delta_ee_actions(actions, block)
        else:
            self._execute_joint_actions(actions, block)

    def _execute_joint_actions(self, actions: np.ndarray, block: bool = False):
        """Execute joint position actions."""
        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            self.robot_client.robot_move_to_joint_positions(positions = actions[:7], time_to_go = 1.0)
            self.robot_client.gripper_grasp(width=0.085, speed=self.gripper_speed, force=self.gripper_force, epsilon_inner=0.0801, epsilon_outer=0.0801)
            logging.info("[STATE] Robot reached initial pose.")

        else:
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()

                joint_positions = action[:7]
                # Move robot
                self.robot_client.robot_update_desired_joint_positions(joint_positions)
                # Control gripper
                gripper_command = 0 if action[7] < self.close_threshold else 1
                if self.gripper_reverse:
                    gripper_command = 1 - gripper_command
                self.robot_client.gripper_goto(width=gripper_command*0.0801, speed=self.gripper_speed, force=self.gripper_force, epsilon_inner=0.0801, epsilon_outer=0.0801)
                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    def _execute_delta_ee_actions(self, actions: np.ndarray, block: bool = False):
        """Execute delta end-effector actions.
        
        Action format: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        The deltas are applied to the current end-effector pose using proper rotation composition.
        """
        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            # For block mode, actions are absolute ee pose
            initial_pose = actions[:6]
            self.robot_client.robot_move_to_ee_pose(
                position=initial_pose[:3],
                orientation=initial_pose[3:6],
                time_to_go=1.0
            )
            self.robot_client.gripper_grasp(width=0.085, speed=self.gripper_speed, force=self.gripper_force, epsilon_inner=0.0801, epsilon_outer=0.0801)
            logging.info("[STATE] Robot reached initial pose.")
        else:
            # Get current ee pose as base
            current_ee_pose = self.robot_client.robot_get_ee_pose()  # [x, y, z, rx, ry, rz]
            current_pos = current_ee_pose[:3].copy()
            current_euler = current_ee_pose[3:6].copy()
            print(f"[STATE] Initial EE Pose: {current_ee_pose}")
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()

                # Delta ee action: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
                delta_pos = action[:3] * self.ee_action_scale
                delta_euler = action[3:6] * self.ee_action_scale
                print(f"[STATE] Delta EE: {delta_pos}, {delta_euler}")
                # Apply delta position (can be added directly)
                target_pos = current_pos + delta_pos
                
                # Apply delta rotation using rotation matrices
                target_euler = apply_delta_rotation(current_euler, delta_euler)
                
                # Combine into target pose
                target_pose = np.concatenate([target_pos, target_euler])
                
                # Move robot to target ee pose
                self.robot_client.robot_update_desired_ee_pose(target_pose)
                print(f"[STATE] Target EE Pose: {target_pose}")
                # Update current pose for next iteration
                current_pos = target_pos.copy()
                current_euler = target_euler.copy()

                # Control gripper
                gripper_command = 0 if action[6] < self.close_threshold else 1
                if self.gripper_reverse:
                    gripper_command = 1 - gripper_command
                self.robot_client.gripper_goto(width=gripper_command*0.0801, speed=self.gripper_speed, force=self.gripper_force, epsilon_inner=0.0801, epsilon_outer=0.0801)
                
                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect robot, cameras, and print state."""
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        # self.execute_actions(self.initial_pose, block=True) # move to initial pose
        self.robot_client.robot_move_to_joint_positions(positions=self.initial_pose, time_to_go=5.0)
        
        # Start appropriate control mode based on action_mode
        if self.action_mode == "delta_ee":
            logging.info("[STATE] Starting Cartesian impedance control for delta_ee mode...")
            self.robot_client.robot_start_cartesian_impedance_control()
        else:
            logging.info("[STATE] Starting joint impedance control...")
            self.robot_client.robot_start_joint_impedance_control()
        
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state: {obs.keys()}")
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        logging.info("Warming up the model")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                start_time = time.perf_counter()
                obs = self.get_obs_state()
                result = policy.infer(obs)
                self.execute_actions(result["actions"])
                self.recorder.submit_actions(result["actions"][:self.action_horizon], infer_time, obs["prompt"])
                self.recorder.submit_obs(obs)
                end_time = time.perf_counter()
                logging.info(f"[STATE] Inference loop rate: {1 / (end_time - start_time):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detec ted. Saving recorded videos before exiting...")

        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")

        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_franka_pi.yaml"
    inference = Inference(config_path)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
