"""AME package for Unitree G1 on mjlab."""

from __future__ import annotations

from pathlib import Path

__all__ = ["__version__", "assert_installed_mjlab"]

__version__ = "0.1.0"

_PACKAGE_ROOT = Path(__file__).resolve()
_WORKSPACE_ROOT = _PACKAGE_ROOT.parents[3]
_REFERENCE_MJLAB_ROOT = _WORKSPACE_ROOT / "mjlab"
_REFERENCE_MJLAB_SRC = _REFERENCE_MJLAB_ROOT / "src"


def assert_installed_mjlab(mjlab_module) -> None:
  """Reject the local reference mjlab repo at runtime."""

  mjlab_path = Path(mjlab_module.__file__).resolve()
  if _REFERENCE_MJLAB_ROOT in mjlab_path.parents or _REFERENCE_MJLAB_SRC in mjlab_path.parents:
    raise RuntimeError(
      "AME must import the installed `mjlab` package, not the local reference repo."
    )
