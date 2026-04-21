import dataclasses
import os
import jax

from openpi.models import model as _model
from openpi.policies import nero_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi05_droid_finetune_nero")
# Ensure you replace this path with your real checkpoint directory
checkpoint_dir = "/path/to/your/nero/checkpoint"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = nero_policy.make_nero_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)