"""Collision scratch capacity for the full G1 community course."""

from ame_g1.config.variants import AmeTaskSpec


def configure_collision_capacity(task_spec: AmeTaskSpec) -> dict[str, int]:
  """Configure and report capacities before environment creation or compilation."""
  from mujoco_warp._src import collision_convex, collision_flex, types

  modules = (types, collision_convex, collision_flex)
  if task_spec.variant in ("g1", "ame2-gaze") and task_spec.phase == "base":
    # mujoco-warp 3.11.0 has no public EPA horizon capacity setting. Both
    # community-course tasks exceeded 24. Capacity 64 cleared the saved G1
    # failure; the Gaze failure did not retain a replayable physics snapshot.
    for module in modules:
      module.MJ_MAX_EPAHORIZON = 64

  return {module.__name__: int(module.MJ_MAX_EPAHORIZON) for module in modules}
