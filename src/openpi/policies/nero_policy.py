import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_nero_example() -> dict:
    """Creates a random input example for a NERO policy.

    The recommended NERO schema is:
    - proprio state: 14D
    - action: 14D
    - cameras: configurable multi-camera RGB inputs
    """
    return {
        # 14D = left 6D EE pose + right 6D EE pose + 2 gripper command scalars.
        "observation/state": np.random.rand(14),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NeroInputs(transforms.DataTransformFn):
    """Convert NERO observations into the common model input format.

    This is the robot-facing interface that should be adapted to the real NERO
    observation schema during training and inference.

    Recommended mapping:
    - `state`: 14D proprio vector
    - `image`: configurable RGB camera inputs
    - `actions`: 14D control vector during training
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    # Source keys in the raw NERO observation dict.
    state_key: str = "observation/state"
    base_image_key: str = "observation/image"
    left_wrist_image_key: str = "observation/wrist_image"
    right_wrist_image_key: str | None = None
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data[self.base_image_key])
        left_wrist_image = _parse_image(data[self.left_wrist_image_key])

        if self.right_wrist_image_key is not None and self.right_wrist_image_key in data:
            right_wrist_image = _parse_image(data[self.right_wrist_image_key])
            right_wrist_mask = np.True_
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data[self.state_key],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if self.prompt_key in data:
            prompt = data[self.prompt_key]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class NeroOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the NERO action format.

    The default NERO action space is 14D:
    - 12D dual-arm end-effector delta control
    - 2D continuous gripper commands
    """

    # Number of action dimensions to return to the robot.
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}