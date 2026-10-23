"""Convert an official Gaze actor to native joint order for public play/eval.

This does not migrate the Isaac Lab critic or optimizer. Source joint names must
come from the source robot's ordering; no alphabetical or positional guess is made.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
import yaml

from ame_g1.config.g1.ame2_env_cfg import make_ame2_gaze_teacher_env_cfg
from ame_g1.config.g1.ame2_rl_cfg import GazeTeacherActorCfg
from ame_g1.learning.ame2_models import AME2GazeTeacherModel


def convert_actor_weights(state, source_joints, native_joints):
  if (len(source_joints) != 23 or len(set(source_joints)) != 23
      or len(native_joints) != 23 or set(source_joints) != set(native_joints)):
    raise ValueError("Official and native joint orders must contain the same 23 unique joint names")
  order = torch.tensor([source_joints.index(name) for name in native_joints])
  converted = {"network." + key: value.clone() for key, value in state.items() if key != "log_std"}
  # prop = velocity/gravity (9), joint position (23), velocity (23), last action (23), command (4).
  columns = torch.arange(82)
  for start in (9, 32, 55):
    columns[start:start + 23] = start + order
  converted["network.prop_encoder.net.0.weight"] = state["prop_encoder.net.0.weight"][:, columns]
  for key in ("output_mlp.net.4.weight", "output_mlp.net.4.bias"):
    converted["network." + key] = state[key][order]
  converted["distribution.log_std_param"] = state["log_std"][order]
  return converted


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--joint-order", type=Path, required=True,
                      help="JSON with joint_names in the official policy's input/action order")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  source_config = yaml.safe_load((args.checkpoint.parent / "params/agent.yaml").read_text())
  actor_config = source_config["actor"]
  native_config = asdict(GazeTeacherActorCfg())
  if (actor_config["class_name"] != "AME2GazeModel"
      or actor_config["noise_std_type"] != "log"
      or source_config["empirical_normalization"] is not None
      or source_config["clip_actions"] is not None):
    raise ValueError("Expected the official Gaze teacher with log std and no observation/action normalization")
  for key in ("num_heads", "crop_x", "crop_y", "roi_coef", "roi_margin"):
    if actor_config[key] != native_config[key]:
      raise ValueError(f"Official actor setting {key} differs from the registered Gaze task")
  source_joints = json.loads(args.joint_order.read_text())["joint_names"]
  cfg = make_ame2_gaze_teacher_env_cfg(play=True)
  native_joints = list(cfg.actions["joint_pos"].actuator_names)
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  state = convert_actor_weights(checkpoint["actor_state_dict"], source_joints, native_joints)
  obs = TensorDict({"teacher_prop": torch.zeros(1, 82),
                   "teacher_mapping": torch.zeros(1, 26, 15, 3)}, batch_size=[1])
  native_config.pop("class_name")
  actor = AME2GazeTeacherModel(obs, {"actor": list(obs.keys())}, "actor", 23, **native_config)
  actor.load_state_dict(state, strict=True)
  if not all(torch.isfinite(value).all() for value in state.values()):
    raise ValueError("Official actor contains non-finite weights")
  provenance = {
    "source_checkpoint": str(args.checkpoint.resolve()),
    "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "source_joint_order_sha256": hashlib.sha256(args.joint_order.read_bytes()).hexdigest(),
    "source_joint_names": source_joints, "native_joint_names": native_joints,
    "inference_only": True, "task": "AME2-G1-Gaze-Teacher",
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as file:
    torch.save({"actor_state_dict": state, "iter": checkpoint["iter"],
                "infos": {}, "ame2_import": provenance}, file)
  print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
  main()
