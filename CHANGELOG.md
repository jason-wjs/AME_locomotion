# Changelog

## Unreleased — mjlab 1.6 compatibility

- Lock the simulation stack to mjlab 1.6.0; retain the `g1` default, `paper`
  baseline, `base/finetune` phases, and global-context option.
- Recognize global-context models from the actual RSL-RL actor checkpoint state.
- Fix global-context TorchScript export without changing model parameter layouts.
- Preserve the previous command mixture and raycast geometry groups instead of
  inheriting new upstream training defaults; omit the unused foot-height sensor.
- Honor the upstream `--log-root` option and save agent configuration before
  RSL-RL mutates it during construction.
- Document the stable baseline, AME-2 implementation differences, compatibility
  boundaries, and maintenance workflow; add a focused CPU CI job.

See [the migration inventory](docs/migration-mjlab-1.6.md) for scope and validation.
This entry is not a release or a claim of training-performance equivalence.
