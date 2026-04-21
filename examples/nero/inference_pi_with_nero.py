import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _as_float32(vec: Any, expected_dim: int, *, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"{name} must be {expected_dim}D, got shape {arr.shape}")
    return arr


class NeroRobotAdapter:
    """Hardware adapter for NERO.

    Replace this class with your real NERO SDK calls.
    Required observation keys returned by `get_observation()`:
    - left_ee_pose: [6], order x,y,z,rz,ry,rx
    - right_ee_pose: [6], order x,y,z,rz,ry,rx
    - left_gripper_cmd: scalar (continuous)
    - right_gripper_cmd: scalar (continuous)
    - images: dict[str, HxWx3 uint8]
    """

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def move_to_home(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        raise NotImplementedError("Implement NERO observation readout in NeroRobotAdapter.get_observation().")

    def apply_action(
        self,
        *,
        left_delta_ee: np.ndarray,
        right_delta_ee: np.ndarray,
        left_gripper_cmd: float,
        right_gripper_cmd: float,
    ) -> None:
        raise NotImplementedError("Implement NERO action execution in NeroRobotAdapter.apply_action().")


class DummyNeroRobotAdapter(NeroRobotAdapter):
    """Dry-run adapter for validating mapping without hardware."""

    def connect(self) -> None:
        logging.info("[DUMMY] Connected to fake NERO adapter")

    def disconnect(self) -> None:
        logging.info("[DUMMY] Disconnected from fake NERO adapter")

    def move_to_home(self) -> None:
        logging.info("[DUMMY] Move to home pose")

    def get_observation(self) -> dict[str, Any]:
        img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        return {
            "left_ee_pose": np.zeros(6, dtype=np.float32),
            "right_ee_pose": np.zeros(6, dtype=np.float32),
            "left_gripper_cmd": 0.0,
            "right_gripper_cmd": 0.0,
            "images": {
                "head_image": img,
                "left_wrist_image": img,
                "right_wrist_image": img,
            },
        }

    def apply_action(
        self,
        *,
        left_delta_ee: np.ndarray,
        right_delta_ee: np.ndarray,
        left_gripper_cmd: float,
        right_gripper_cmd: float,
    ) -> None:
        logging.info(
            "[DUMMY] action left=%s right=%s left_gripper=%.3f right_gripper=%.3f",
            np.round(left_delta_ee, 4).tolist(),
            np.round(right_delta_ee, 4).tolist(),
            left_gripper_cmd,
            right_gripper_cmd,
        )


class NeroInference:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]
        self.model_config = _config.get_config(model_cfg["name"])
        self.checkpoint_dir = Path(model_cfg["checkpoint_dir"])

        run_cfg = cfg["run"]
        self.action_fps = run_cfg.get("action_fps", 20)
        self.action_horizon = run_cfg.get("action_horizon", 10)
        self.max_steps = run_cfg.get("max_steps", 0)
        self.default_prompt = run_cfg.get("task_description", "")

        image_cfg = cfg["image"]
        self.resize_height = image_cfg.get("height", 224)
        self.resize_width = image_cfg.get("width", 224)
        self.base_camera_key = image_cfg.get("base_camera_key", "head_image")
        self.left_wrist_camera_key = image_cfg.get("left_wrist_camera_key", "left_wrist_image")
        self.right_wrist_camera_key = image_cfg.get("right_wrist_camera_key", "right_wrist_image")

        gripper_cfg = cfg.get("gripper", {})
        self.use_binary_mode = gripper_cfg.get("use_binary_mode", False)
        self.close_threshold = float(gripper_cfg.get("close_threshold", 0.05))
        self.min_cmd = float(gripper_cfg.get("min_cmd", 0.0))
        self.max_cmd = float(gripper_cfg.get("max_cmd", 1.0))

        self.dry_run = bool(cfg.get("dry_run", True))
        self.robot = DummyNeroRobotAdapter() if self.dry_run else NeroRobotAdapter()

    def _transfer_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Map NERO raw observation into policy input.

        State semantics (14D):
        left_ee_pose[0:6] + right_ee_pose[0:6] + left_gripper + right_gripper
        where EE pose order is x,y,z,rz,ry,rx.
        """
        left_ee = _as_float32(raw_obs["left_ee_pose"], 6, name="left_ee_pose")
        right_ee = _as_float32(raw_obs["right_ee_pose"], 6, name="right_ee_pose")
        left_gripper = np.asarray([float(raw_obs["left_gripper_cmd"])], dtype=np.float32)
        right_gripper = np.asarray([float(raw_obs["right_gripper_cmd"])], dtype=np.float32)

        state = np.concatenate((left_ee, right_ee, left_gripper, right_gripper), axis=0)

        images = raw_obs["images"]
        base_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[self.base_camera_key], self.resize_height, self.resize_width)
        )
        left_wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[self.left_wrist_camera_key], self.resize_height, self.resize_width)
        )

        mapped = {
            "observation/state": state,
            "observation/image": base_image,
            "observation/wrist_image": left_wrist,
            "prompt": self.default_prompt,
        }

        # Optional right wrist image path for NeroInputs.
        if self.right_wrist_camera_key in images:
            mapped["observation/right_wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images[self.right_wrist_camera_key], self.resize_height, self.resize_width)
            )

        return mapped

    def _decode_policy_action(self, action_vec: np.ndarray) -> dict[str, Any]:
        """Decode 14D policy action into NERO execution command.

        Action semantics (14D):
        - [0:6]  left_delta_ee_pose  (x,y,z,rx,ry,rz)
        - [6:12] right_delta_ee_pose (x,y,z,rx,ry,rz)
        - [12]   left_gripper_cmd
        - [13]   right_gripper_cmd
        """
        action_vec = _as_float32(action_vec, 14, name="policy_action")

        left_delta_ee = action_vec[:6]
        right_delta_ee = action_vec[6:12]

        left_gripper_cmd = float(np.clip(action_vec[12], self.min_cmd, self.max_cmd))
        right_gripper_cmd = float(np.clip(action_vec[13], self.min_cmd, self.max_cmd))

        if self.use_binary_mode:
            left_gripper_cmd = 0.0 if left_gripper_cmd < self.close_threshold else 1.0
            right_gripper_cmd = 0.0 if right_gripper_cmd < self.close_threshold else 1.0

        return {
            "left_delta_ee": left_delta_ee,
            "right_delta_ee": right_delta_ee,
            "left_gripper_cmd": left_gripper_cmd,
            "right_gripper_cmd": right_gripper_cmd,
        }

    def run(self) -> None:
        self.robot.connect()
        self.robot.move_to_home()

        policy = _policy_config.create_trained_policy(
            self.model_config,
            self.checkpoint_dir,
        )

        # Warmup
        warmup_obs = self._transfer_observation(self.robot.get_observation())
        policy.infer(warmup_obs)

        step = 0
        try:
            while True:
                t0 = time.perf_counter()

                raw_obs = self.robot.get_observation()
                obs = self._transfer_observation(raw_obs)
                result = policy.infer(obs)

                # Execute up to configured action horizon.
                for action in result["actions"][: self.action_horizon]:
                    cmd = self._decode_policy_action(action)
                    self.robot.apply_action(**cmd)

                step += 1
                if self.max_steps > 0 and step >= self.max_steps:
                    break

                dt = time.perf_counter() - t0
                sleep_t = 1.0 / self.action_fps - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            self.robot.move_to_home()
            self.robot.disconnect()


def main() -> None:
    config_path = Path(__file__).parent / "config" / "cfg_nero_pi.yaml"
    NeroInference(config_path).run()


if __name__ == "__main__":
    main()
