"""AME attention plotting entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro


@dataclass(frozen=True)
class PlotAttentionArgs:
  dump_file: str
  output_dir: str = "artifacts/attention"
  max_frames: int = 16
  env_index: int = 0


def main() -> None:
  args = tyro.cli(PlotAttentionArgs)
  dump = torch.load(Path(args.dump_file).expanduser(), map_location="cpu", weights_only=False)
  output_dir = Path(args.output_dir).expanduser()
  output_dir.mkdir(parents=True, exist_ok=True)

  for frame_idx, step in enumerate(dump["steps"][: args.max_frames]):
    weights = step["attention_weights"]
    if weights is None:
      continue
    points_tensor = step.get("attention_points")
    if points_tensor is None:
      points_tensor = step["terrain_points"]
    points = points_tensor[args.env_index]
    weights = weights[args.env_index]
    x = points[..., 0].reshape(-1)
    y = points[..., 1].reshape(-1)
    z = points[..., 2].reshape(-1)
    w = weights.reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    scatter = ax.scatter(
      x,
      y,
      c=w,
      s=140,
      cmap="Reds",
      edgecolors="black",
      linewidths=0.2,
    )
    ax.set_title(f"Step {frame_idx}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("attention")
    for px, py, pz in zip(x.tolist(), y.tolist(), z.tolist(), strict=False):
      ax.text(px, py, f"{pz:.2f}", fontsize=6, ha="center", va="center")
    output_path = output_dir / f"attention_{frame_idx:04d}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

  print(f"[INFO] Wrote attention plots to {output_dir}")


if __name__ == "__main__":
  main()
