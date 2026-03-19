# $\pi_{0.5}$ for Franka
In this project, I finetune $\pi_{0.5}$-DROID on my own Franka dataset, 
collected by the [lerobot_franka_isoteleop](https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git) project. The dataset is in the format of Lerobot V3.0.

## Env Setup and Installation

### 1. Clone the repository:
```bash
git clone --recurse-submodules https://github.com/Shenzhaolong1330/openpi-franka.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

### 2. Add the dependencies you need:

You can add the dependencies you need in the [pyproject.toml](pyproject.toml) `[project] dependencies` section.
For me I need the following dependencies:
```toml
"pyrealsense2",
"zerorpc",
```

As for [lerobot](https://github.com/huggingface/lerobot), you can add in the `[tool.uv.sources]` section.
The default version of lerobot installed by `openpi` is a very old version and the data format is not compatible with the latest version.
So we need to install a proper version of lerobot manually.
Here I chose the version `da5d2f3e9187fa4690e6667fe8b294cae49016d6` which is the version compatible with data collected by [lerobot_franka_isoteleop](https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git).
Now as I have installed the proper version of lerobot, I can add local path of lerobot in the `[tool.uv.sources]` section.
```toml
lerobot = {path = "/path/to/lerobot"}
```

### 3. Env management

Create a conda environment for this project:
```bash
conda create -n openpi-franka python=3.11
conda activate openpi-franka
```
Install `uv` in the conda environment:
```bash
pip install uv
```
and install the dependencies in the conda environment:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Config Franka Policy

In [franka_policy.py](src/openpi/policies/franka_policy.py), Class `FrankaInputs` and `FrankaOutputs` is defined.
<!-- ```python
@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}
``` -->
The `FrankaInputs` is used to convert the inputs to the model to the expected format.
The `FrankaOutputs` is used to convert the outputs of the model to the expected format.

## Config Franka Data

In [config.py](src/openpi/training/config.py), Class `LeRobotFrankaDataConfig` is defined.
<!-- ```python
@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDataConfig(DataConfigFactory):
    """
    Example data config for custom Franka dataset in LeRobot format created by https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git.
    """
    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for UR5e).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.exterior_image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model(``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `UR5e_policy.py`. You can checkthe detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[franka_policy.FrankaInputs(model_type=model_config.model_type)],
            outputs=[franka_policy.FrankaOutputs()],
        )
        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
``` -->
This class is used to configure the Franka dataset in the format of Lerobot V3.0.
It will be used to process and convert the data to the expected format in the training and inference process.

## Config Franka Trainer

In [config.py](src/openpi/training/config.py), TrainConfig for `pi05_droid_finetune_franka` is defined.

<!-- ```python
TrainConfig(
        name="pi05_droid_finetune_franka",
        model=pi0_config.Pi0Config(
            pi05=True, 
            action_horizon=10, 
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotFrankaDataConfig(
            repo_id="/home/deepcybo/.cache/huggingface/lerobot/shenzhaolong/pick_and_place_merged",
            base_config=DataConfig(prompt_from_task=False, action_sequence_keys=("action",)),
            extra_delta_transform=True,
        ),
        batch_size=32,
        #lr_schedule=_optimizer.CosineDecaySchedule(
        #     warmup_steps=10_000,
        #     peak_lr=5e-5,
        #     decay_steps=1_000_000,
        #     decay_lr=5e-5,
        # ),
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # ema_decay=0.999,
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/deepcybo/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
``` -->
This part configures the model, dataset, and parameters for training, and other hyperparameters.

Remember to set your own datasets in the `repo_id` field, and  `weight_loader` to your own checkpoint.

## `data_loader` Reorder `observation.state` and Compute the State Norm

In my own datasets, the `obs.state` is not only the concatenation of the joint angles and the gripper position, but also contains the velocity of the joints and other information.
To match the format of the pretrained models, we need to reorder the `observation.state` to be `[joint_position, gripper_position]`.

I added the following code to reorder the `observation.state` in the [data_loader.py](src/openpi/training/data_loader.py) `create_torch_dataset()`.
<!-- ```python
# data reorder
    def reorder_state(example, indices):
        state = example['observation.state']
        if max(indices) > len(state) or min(indices) < 1:
            raise IndexError(f"indices {indices} out of range for observation.state of length {len(state)}")
        example['observation.state'] = [state[i-1] for i in indices]
        return example

    obs_indices = os.getenv("OBS_INDICES")
    if obs_indices:
        indices = list(map(int, obs_indices.split(",")))
    else:
        raise ValueError("Environment variable OBS_INDICES not set for reordering observation.state")
    print(f"Reordering state with indices: {indices}")
    print(f"Reordering state with names: {[dataset_meta.names['observation.state'][i-1] for i in indices]}")
    dataset.hf_dataset = dataset.hf_dataset.map(lambda ex: reorder_state(ex, indices=indices))
    print("Finished reordering state.")
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
``` -->

For computing the state norm, we need to set the `OBS_INDICES` environment variable to `1,2,3,4,5,6,7,9` (the indices of the joint positions and gripper position).
```bash
# OBS_INDICES is the indices of the joint positions and gripper position in the observation.state
# --config-name is the name of the config name in the config.py
OBS_INDICES=1,2,3,4,5,6,7,8 uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_franka
```
The computed state norm is saved in the `norm_stats.json` file in the dataset directory.

## Finetune the Model

I collected 50 training trajectories for finetuning the model.

```python
OBS_INDICES=1,2,3,4,5,6,7,8 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_finetune_franka --exp_name=pick_and_place_robotiq_0211 --overwrite
```

## Inference the Model
[inference_pi_with_franka.py](examples/franka/inference_pi_with_franka.py)


## Use Tos
Get data from tos:
```bash
./tosutil cp -r -p 40 -j 50 -nfj 40 tos://c20250510/shenzhaolong/datasets/pick_all_objects_20260208/ workspace/data/robotiq
```

Send model to tos:
```bash
./tosutil cp -r -p 40 -j 50 -nfj 40 /vepfs-mlp2/c20250510/250303034/workspace/openpi-franka/checkpoints/pi05_droid_finetune_franka/pick_and_place_robotiq_0211/55000 tos://c20250510/shenzhaolong/datasets/model/pick_and_place_robotiq_0211/55000
```

Get model from tos:
```bash
tosutil cp -r -p 40 -j 50 -nfj 40 tos://c20250510/shenzhaolong/datasets/model/pick_and_place_robotiq_0211/55000 /home/deepcybo/.cache/openpi/openpi-assets/checkpoints/pi05_droid_finetune_franka/pick_and_place_robotiq
```

tosutil cp -r -p 40 -j 50 -nfj 40 tos://c20250510/shenzhaolong/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /vepfs-mlp2/c20250510/250303034/workspace/starVLA
