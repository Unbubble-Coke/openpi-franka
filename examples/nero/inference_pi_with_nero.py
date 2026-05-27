import openpi.training.checkpoints
import openpi.policies.policy_config
import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config

import logging
import time
import os
import json

print("[DEBUG] Importing pathlib and typing...", flush=True)
from pathlib import Path
from typing import Dict, Any

print("[DEBUG] Importing numpy...", flush=True)
import numpy as np

print("[DEBUG] Importing yaml...", flush=True)
import yaml

print("[DEBUG] Importing scipy Rotation...", flush=True)
from scipy.spatial.transform import Rotation as R

print("[DEBUG] Importing utils and openpi_client...", flush=True)
from utils import FpsCounter
from openpi_client import image_tools
from recorder import Recorder 

print("[DEBUG] Importing lerobot cameras...", flush=True)
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs

# Import the actual client we just copied from the teleop folder
print("[DEBUG] Importing NeroDualArmClient...", flush=True)
from nero_interface_client import NeroDualArmClient

print("[DEBUG] All top-level imports successful!", flush=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 获取当前项目根目录 (openpi-franka)
repo_root = Path(__file__).resolve().parent.parent.parent

def rotvec_to_rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotation_vector).as_matrix()

def rotation_matrix_to_rotvec(rot_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_matrix).as_rotvec()

def apply_delta_rotation(current_rotvec: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Apply delta rotation to current rotation using rotation matrices."""
    current_rot = rotvec_to_rotation_matrix(current_rotvec)
    delta_rot = rotvec_to_rotation_matrix(delta_rotvec)
    # 恢复为你采数据时所用的左乘逻辑
    new_rot = delta_rot @ current_rot
    return rotation_matrix_to_rotvec(new_rot)

def limit_vector_norm(vector: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, bool]:
    if max_norm is None or max_norm <= 0:
        return vector, False

    norm = float(np.linalg.norm(vector))
    if norm <= max_norm:
        return vector, False

    return vector * (max_norm / norm), True

def _zero_like_dims(values: list[float], dim: int, eps: float = 1e-6) -> list[int]:
    arr = np.asarray(values[:dim], dtype=np.float32)
    return np.flatnonzero(np.abs(arr) < eps).tolist()

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
        self.checkpoint_dir = Path(chk_dir) if chk_dir.startswith("/") else repo_root / chk_dir
        
        # Camera config (3 cameras for complete Nero vision setup)
        cam = cfg.get("cameras", {})
        self.left_wrist_cam_serial = str(cam.get("left_wrist_cam_serial", ""))
        self.right_wrist_cam_serial = str(cam.get("right_wrist_cam_serial", ""))
        self.exterior_cam_serial = str(cam.get("exterior_cam_serial", ""))
        self.cam_fps = cam.get("fps", 30)
        self.cam_width = cam.get("width", None)
        self.cam_height = cam.get("height", None)

        video = cfg.get("video", {"fps": 15, "visualize": True})
        self.video_fps = video.get("fps", 15)
        self.visualize = video.get("visualize", True)

        robot = cfg.get("robot", {})
        self.robot_ip = robot.get("ip", "192.168.110.203")
        self.robot_port = robot.get("port", 4242)
        self.initial_left_joints = np.asarray(robot.get("initial_left_joints", np.zeros(7)), dtype=np.float32)
        self.initial_right_joints = np.asarray(robot.get("initial_right_joints", np.zeros(7)), dtype=np.float32)
        self.dry_run = cfg.get("dry_run", False)
        
        run_cfg = cfg.get("run", {})
        self.action_fps = run_cfg.get("action_fps", robot.get("action_fps", 20))
        self.action_horizon = run_cfg.get("action_horizon", robot.get("action_horizon", 10))
        self.state_diag_period_s = run_cfg.get("state_diag_period_s", 2.0)

        gripper = cfg.get("gripper", {})
        self.use_binary_gripper = gripper.get("use_binary_mode", True)
        self.close_threshold = gripper.get("close_threshold", 0.05)
        self.gripper_min_cmd = gripper.get("min_cmd", 0.0)
        self.gripper_max_cmd = gripper.get("max_cmd", 1.0)
        self.gripper_max_width = gripper.get("max_width", 0.1)
        self.gripper_force = gripper.get("gripper_force", 2.0)
        self.gripper_speed = gripper.get("gripper_speed", 0.1)
        self.gripper_reverse = gripper.get("gripper_reverse", False)

        action_mode = cfg.get("action_mode", {})
        self.action_mode = action_mode.get("mode", "delta_ee")
        self.ee_action_scale = action_mode.get("ee_action_scale", 1.0)
        self.max_delta_pos_norm = action_mode.get("max_delta_pos_norm", 0.01)
        self.max_delta_rot_norm = action_mode.get("max_delta_rot_norm", 0.05)
        self.freeze_right_arm_action = action_mode.get("freeze_right_arm_action", False)
        self.freeze_right_gripper_action = action_mode.get(
            "freeze_right_gripper_action",
            self.freeze_right_arm_action,
        )
        self._last_delta_clip_log_time = 0.0
        self._last_state_diag_log_time = 0.0
        self._norm_stats = None

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

    def _check_data_alignment(self, policy):
        """
        在部署模型前进行虚拟数据对齐检测。
        通过输入一组特定的 state (如偏移量 x=10.0)，观察输出的 action。
        验证输出的动作是相对坐标 (delta) 还是绝对坐标 (absolute)。
        """
        logging.info("[CHECK] Running data alignment check for Pi0.5 NERO deployment...")
        try:
            # 构造测试 observation. Training used OBS_INDICES=1..28, so the
            # policy state is the full LeRobot NERO state, not the 14D EE-only view.
            test_obs = {
                "observation/state": np.zeros(28, dtype=np.float32),
                "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/right_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "prompt": "pick and place",
            }
            # 设置初始位置
            test_obs["observation/state"][14] = 10.0
            test_obs["observation/state"][20] = 20.0
            
            result = policy.infer(test_obs)
            actions = result.get("actions")
            
            if actions is not None:
                # 检查输出的 actions 的量级。如果输出是绝对方位，它会偏向于 10.0；如果是增量，它会偏向于 0.0。
                val_x = float(actions[0, 0])
                if abs(val_x - 10.0) < 5.0:
                    logging.warning("[WARN] Data alignment check: Action seems to be ABSOLUTE mode (value near 10, got %.4f). But inference script uses servo_p_OL with delta=True. Verify action_mode!", val_x)
                elif abs(val_x) < 2.0:
                    logging.info("[CHECK] Data alignment check passed: Action seems to be DELTA mode (value near 0, got %.4f). Compatible with delta_ee matching.", val_x)
                else:
                    logging.warning("[WARN] Data alignment check: Action value %.4f is unexpected for state x=10.0. Please verify the scale.", val_x)
        except Exception as e:
            logging.warning("[CHECK] Failed to test data alignment: %s", e)

    def _check_nero_norm_stats(self):
        try:
            data_config = self.model_config.data.create(self.model_config.assets_dirs, self.model_config.model)
            asset_id = data_config.asset_id
            if asset_id is None:
                return

            stats_path = self.checkpoint_dir / "assets" / asset_id / "norm_stats.json"
            if not stats_path.exists():
                logging.warning("[CHECK] Norm stats not found at %s", stats_path)
                return

            stats = json.loads(stats_path.read_text())["norm_stats"]
            self._norm_stats = stats
            action_std = stats["actions"]["std"]
            state_std = stats["state"]["std"]
            expected_state_dim = 28
            expected_action_dim = 14

            if len(state_std) < expected_state_dim or len(action_std) < expected_action_dim:
                logging.warning(
                    "[CHECK] Norm stats look inconsistent with training schema. "
                    "asset_id=%s, state std len=%d, action std len=%d. "
                    "Expected at least %dD state and %dD action.",
                    asset_id,
                    len(state_std),
                    len(action_std),
                    expected_state_dim,
                    expected_action_dim,
                )
            else:
                logging.info(
                    "[CHECK] Norm stats schema: state std len=%d, action std len=%d. "
                    "Zero std action dims in first 14=%s, zero std state dims in first 28=%s.",
                    len(state_std),
                    len(action_std),
                    _zero_like_dims(action_std, expected_action_dim),
                    _zero_like_dims(state_std, expected_state_dim),
                )
        except Exception as e:
            logging.warning("[CHECK] Failed to validate NERO norm stats: %s", e)

    def _log_state_diagnostics(self, obs: Dict[str, Any], *, force: bool = False):
        if self._norm_stats is None:
            return

        now = time.perf_counter()
        if not force and now - self._last_state_diag_log_time < self.state_diag_period_s:
            return
        self._last_state_diag_log_time = now

        state = np.asarray(obs["observation/state"], dtype=np.float32)
        stats = self._norm_stats["state"]
        mean = np.asarray(stats["mean"][: state.shape[0]], dtype=np.float32)
        std = np.asarray(stats["std"][: state.shape[0]], dtype=np.float32)
        q01 = np.asarray(stats.get("q01", mean)[: state.shape[0]], dtype=np.float32)
        q99 = np.asarray(stats.get("q99", mean)[: state.shape[0]], dtype=np.float32)
        std_safe = np.where(std < 1e-4, 1.0, std)
        zscore = (state - mean) / std_safe

        names = [
            "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7",
            "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7",
            "left_x", "left_y", "left_z", "left_rx_saved", "left_ry", "left_rz_saved",
            "right_x", "right_y", "right_z", "right_rx_saved", "right_ry", "right_rz_saved",
            "left_grip", "right_grip",
        ]
        key_dims = [14, 15, 16, 17, 18, 19, 26]
        parts = []
        for idx in key_dims:
            if idx >= state.shape[0]:
                continue
            out_of_range = state[idx] < q01[idx] or state[idx] > q99[idx]
            marker = "!" if out_of_range else ""
            parts.append(
                f"{names[idx]}={state[idx]:+.4f} "
                f"z={zscore[idx]:+.2f} "
                f"q01/q99=[{q01[idx]:+.4f},{q99[idx]:+.4f}]{marker}"
            )
        logging.info("[STATE_DIAG] %s", " | ".join(parts))

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
            self.robot_client = None

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        if self.dry_run:
            logging.info("[DUMMY] Dry run: Skipping camera connection.")
            return

        try:
            logging.info("\n===== [CAMERAS] Connecting to Realsense Cameras =====")
            configs = {}
            kw = {"fps": self.cam_fps, "color_mode": ColorMode.RGB}
            if self.cam_width and self.cam_height:
                kw["width"] = self.cam_width
                kw["height"] = self.cam_height

            if self.exterior_cam_serial:
                configs["exterior_image"] = RealSenseCameraConfig(serial_number_or_name=self.exterior_cam_serial, **kw)
            if self.left_wrist_cam_serial:
                configs["left_wrist_image"] = RealSenseCameraConfig(serial_number_or_name=self.left_wrist_cam_serial, **kw)
            if self.right_wrist_cam_serial:
                configs["right_wrist_image"] = RealSenseCameraConfig(serial_number_or_name=self.right_wrist_cam_serial, **kw)

            if configs:
                self.cameras = make_cameras_from_configs(configs)
                for name, cam in self.cameras.items():
                    cam.connect()
                logging.info(f"[CAMERAS] Connected: {list(configs.keys())}")
            else:
                logging.error("[CAMERAS] No cameras configured in config file")
        except Exception as e:
            logging.error(f"===== [ERROR] Failed to connect to cameras: {e} =====")

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Training used OBS_INDICES=1..28 on the LeRobot NERO dataset:
        # [left joints(7), right joints(7), left EE(6), right EE(6), left gripper, right gripper].
        state = np.concatenate((
            np.asarray(obs["left_joint_positions"], dtype=np.float32),
            np.asarray(obs["right_joint_positions"], dtype=np.float32),
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
            obs["left_joint_positions"] = self.robot_client.left_robot_get_joint_positions()
            obs["right_joint_positions"] = self.robot_client.right_robot_get_joint_positions()
            # 兼容模型训练时错误的保存顺序 [x, y, z, rz, ry, rx]，把物理 [rx, ry, rz] 进行第3、5位交换
            obs["left_ee_pose"] = self.robot_client.left_robot_get_ee_pose()[[0, 1, 2, 5, 4, 3]]
            obs["right_ee_pose"] = self.robot_client.right_robot_get_ee_pose()[[0, 1, 2, 5, 4, 3]]
            
            l_grip_width = self.robot_client.left_gripper_get_state().get("width", 0.0)
            l_grip_state = max(0.0, min(1.0, l_grip_width / self.gripper_max_width))
            obs["left_gripper_position"] = 0.0 if l_grip_state < self.close_threshold else 1.0
            
            r_grip_width = self.robot_client.right_gripper_get_state().get("width", 0.0)
            r_grip_state = max(0.0, min(1.0, r_grip_width / self.gripper_max_width))
            obs["right_gripper_position"] = 0.0 if r_grip_state < self.close_threshold else 1.0
        else:
            obs["left_joint_positions"] = np.zeros(7, dtype=np.float32)
            obs["right_joint_positions"] = np.zeros(7, dtype=np.float32)
            obs["left_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["right_ee_pose"] = np.zeros(6, dtype=np.float32)
            obs["left_gripper_position"] = 0.0
            obs["right_gripper_position"] = 0.0

        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()
        else:
            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            obs["exterior_image"] = img
            obs["left_wrist_image"] = img
            obs["right_wrist_image"] = img

        obs["prompt"] = self.task_description

        return self._transfer_obs_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def _prepare_actions_for_execution(self, actions: np.ndarray) -> np.ndarray:
        exec_actions = np.asarray(actions, dtype=np.float32).copy()
        if self.freeze_right_arm_action:
            exec_actions[..., 6:12] = 0.0
        if self.freeze_right_gripper_action:
            exec_actions[..., 13] = 1.0
        return exec_actions

    def execute_actions(self, actions: np.ndarray):
        if self.dry_run:
            logging.info("[DUMMY] Executed 14D Action Block.")
            return

        if self.robot_client is None:
            return

        if self.action_mode == "delta_ee":
            self._execute_delta_ee_actions(self._prepare_actions_for_execution(actions))
        else:
            logging.error(f"[ERROR] Unsupported action mode {self.action_mode} for NERO.")

    def _limit_delta_pose(self, delta_pose: np.ndarray, arm_name: str) -> np.ndarray:
        pos, pos_clipped = limit_vector_norm(delta_pose[:3], self.max_delta_pos_norm)
        rot, rot_clipped = limit_vector_norm(delta_pose[3:6], self.max_delta_rot_norm)

        if pos_clipped or rot_clipped:
            now = time.perf_counter()
            if now - self._last_delta_clip_log_time > 1.0:
                logging.warning(
                    "[SAFETY] Clipped %s delta pose. "
                    "pos_norm %.4f -> %.4f, rot_norm %.4f -> %.4f",
                    arm_name,
                    np.linalg.norm(delta_pose[:3]),
                    np.linalg.norm(pos),
                    np.linalg.norm(delta_pose[3:6]),
                    np.linalg.norm(rot),
                )
                self._last_delta_clip_log_time = now

        return np.concatenate([pos, rot])

    def _gripper_action_to_width(self, grip_action: float) -> float:
        if self.use_binary_gripper:
            grip_cmd = 0.0 if grip_action < self.close_threshold else 1.0
        else:
            grip_cmd = float(np.clip(grip_action, self.gripper_min_cmd, self.gripper_max_cmd))
            denom = self.gripper_max_cmd - self.gripper_min_cmd
            if abs(denom) > 1e-6:
                grip_cmd = (grip_cmd - self.gripper_min_cmd) / denom

        if self.gripper_reverse:
            grip_cmd = 1.0 - grip_cmd

        return float(np.clip(grip_cmd, 0.0, 1.0) * self.gripper_max_width)

    def _execute_delta_ee_actions(self, actions: np.ndarray):
        """Execute delta end-effector actions on dual arm.
        Action format (14D): [l_dx, l_dy, l_dz, l_drx, l_dry, l_drz,
                              r_dx, r_dy, r_dz, r_drx, r_dry, r_drz,
                              l_grip, r_grip]"""
        for action in actions[:self.action_horizon]:
            start_time = time.perf_counter()

            # ServoP_OL is sent in relative pose mode, so pass the policy delta directly
            # The model's action tensor was trained with the correct physical order [dx, dy, dz, drx, dry, drz]
            d_l_pos, d_l_rot = action[0:3] * self.ee_action_scale, action[3:6] * self.ee_action_scale
            d_r_pos, d_r_rot = action[6:9] * self.ee_action_scale, action[9:12] * self.ee_action_scale

            delta_l_pose = np.concatenate([d_l_pos, d_l_rot])
            delta_r_pose = np.concatenate([d_r_pos, d_r_rot])
            delta_l_pose = self._limit_delta_pose(delta_l_pose, "left_robot")
            delta_r_pose = self._limit_delta_pose(delta_r_pose, "right_robot")

            self.robot_client.servo_p_OL("left_robot", delta_l_pose, delta=True)
            self.robot_client.servo_p_OL("right_robot", delta_r_pose, delta=True)

            # Control grippers
            l_grip_width = self._gripper_action_to_width(action[12])
            r_grip_width = self._gripper_action_to_width(action[13])

            self.robot_client.left_gripper_goto(width=l_grip_width, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=r_grip_width, force=self.gripper_force)

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        logging.info("========== Starting Inference Pipeline ==========")
        print("[DEBUG] Connecting to robot...", flush=True)
        self.connect_robot()
        print("[DEBUG] Connecting to cameras...", flush=True)
        self.connect_cameras()

        if self.robot_client:
            print("[DEBUG] Moving robots to initial joints...", flush=True)
            if np.any(self.initial_left_joints):
                self.robot_client.left_robot_move_to_joint_positions(self.initial_left_joints)
            if np.any(self.initial_right_joints):
                self.robot_client.right_robot_move_to_joint_positions(self.initial_right_joints)

            print("[DEBUG] Syncing ServoP_OL references...", flush=True)
            left_pose = self.robot_client.left_robot_get_ee_pose()
            right_pose = self.robot_client.right_robot_get_ee_pose()
            self.robot_client.servo_p_OL("left_robot", left_pose, delta=False)
            self.robot_client.servo_p_OL("right_robot", right_pose, delta=False)

            print("[DEBUG] Opening grippers...", flush=True)
            self.robot_client.left_gripper_goto(width=self.gripper_max_width, force=self.gripper_force)
            self.robot_client.right_gripper_goto(width=self.gripper_max_width, force=self.gripper_force)
        
        print("[DEBUG] Fetching first observation state...", flush=True)
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation mapped keys: {obs.keys()}")
        
        print(f"[DEBUG] Creating trained policy from {self.checkpoint_dir}...", flush=True)
        self._check_nero_norm_stats()
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        self._log_state_diagnostics(obs, force=True)
        self._check_data_alignment(policy)
        logging.info("Warming up the model...")
        start = time.time()
        print("[DEBUG] Running policy warmup...", flush=True)
        policy.infer(obs)
        print("[DEBUG] Warmup finished!", flush=True)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                t0 = time.perf_counter()
                obs = self.get_obs_state()
                self._log_state_diagnostics(obs)
                result = policy.infer(obs)
                exec_actions = self._prepare_actions_for_execution(result["actions"])

                print(f"[ACTION] Raw inferred actions (horizon={self.action_horizon}):\n{result['actions'][:self.action_horizon]}")
                logging.info(f"[ACTION] Raw inferred actions (horizon={self.action_horizon}):\n{result['actions'][:self.action_horizon]}")
                if self.freeze_right_arm_action or self.freeze_right_gripper_action:
                    print(f"[ACTION] Executed actions after freeze mask:\n{exec_actions[:self.action_horizon]}")
                    logging.info(f"[ACTION] Executed actions after freeze mask:\n{exec_actions[:self.action_horizon]}")

                self.execute_actions(exec_actions)
                self.recorder.submit_actions(exec_actions[:self.action_horizon], infer_time, obs.get("prompt", ""))
                self.recorder.submit_obs(obs)
                
                logging.info(f"[STATE] Loop rate: {1 / (time.perf_counter() - t0):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detected. Stopping.")
        except Exception as e:
            logging.error(f"[ERROR] Loop error: {e}")
            raise e
            
        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
                
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

        finally:
            if self.robot_client:
                self.robot_client.close()
            if self.cameras:
                for name, cam in self.cameras.items():
                    cam.disconnect()

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_nero_pi.yaml"
    inference = Inference(config_path)
    inference.run()

if __name__ == "__main__":
    main()
