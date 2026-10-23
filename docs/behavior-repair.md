# Behavior repair: AME on mjlab 1.6.0

Status: diagnostic training only; behavior acceptance has not passed. The stable
baseline and default algorithm have not changed. Work is isolated on
`fix/behavior-20261018`.

## Confirmed defects and scope

| Area | Evidence | Correction |
| --- | --- | --- |
| G1 single-support reward | Landing/double support was confused with sustained single support | Use current air/contact times, cap at 0.6 s and gate on XY translation |
| G1 velocity reward | Full-body rotation mixed pitch/roll into a yaw-frame target | Rotate world velocity by heading only |
| G1 observation scales | Community angular/joint velocity scales 0.2/0.05 were missing, with no normalizer | Restore the four actor/critic scales |
| G1 commands | Native XY commands and time-based expansion replaced the community forward-speed/heading distribution | Restore x=[0,1.5], y=0, heading commands and 10 s resampling; retain terrain curriculum |
| G1 terrain scan | 16–28 of 693 rays hit the robot's visual geometry; clipped height errors reached 1.15 m | Scan terrain group 0 only |
| G1 transient body contact | Last sensor sample missed an impact within the history window | Use the maximum force norm over history |
| Play reset | 15/16 resets placed the robot at an old origin; maximum mismatch 31.0 m | Select terrain before placing the robot |
| Play terrain labels | 19/20 random tiles did not match the family inferred from their column | Use structured family columns and difficulty rows, while still randomly selecting tiles at reset |
| Evaluation | Interactive horizon, preterminal positions, and summed tracking error could mislead | Use bounded episodes, true terminal state, displacement from actual start, and mean tracking error; support AME2 goal success |
| AME2 stumble | MuJoCo reports force exerted by the foot; normal support has negative Z and was counted as stumbling | Negate Z locally before comparing horizontal and vertical forces |
| Community G1 encoder | Released weights require Conv–ReLU–BatchNorm, but the migration used Conv–ELU | Restore the community local encoder for G1 with and without global context |
| Community G1 controls | Default posture, PD gains, armature, effort limits and action scaling differed from the source | Restore the community control recipe on the existing mjlab geometry |
| Community G1 scanner frame | The source scans from torso_link, while the native task used pelvis | Use torso_link for the G1 variant, retaining yaw alignment and terrain-only rays |
| G1 radial beams | The square-only generator extends eight beams beyond an 8×4 m tile; the configured width ordering also makes beams widen with difficulty | Generate beams against both rectangular bounds and narrow from 0.18 to 0.12 m |

The G1 input changes require fresh training. Old checkpoints should be evaluated
with the source/configuration that produced them; loading them under the new
observation scales is not a fair policy comparison. Earlier native G1 checkpoints
with the ELU encoder require their original source, including commit `043750d`;
they cannot load into the restored ReLU+BatchNorm architecture. No silent
compatibility fallback is introduced. The released community checkpoint was
converted separately for diagnostics using explicit joint, observation and map
axis permutations; its optimizer was not migrated.

## Evidence so far

Experiments are stored outside tracked source under
`artifacts/repair_20261018/` on the repair checkout. Every training run records its
commit, command, checkpoint and GPU telemetry. Earlier runs used GPU0 and later
shared GPU0/GPU1. Current experiments use only `tele_dev_2gpu_1` GPU1
(`GPU-a99655ac-3056-dcbc-c7d6-097aebad85e1`), with one GPU training or evaluation
process at a time. GPU0 is left untouched.

- G1 original versus single-support-only repair: 500 updates each, 4096
  environments, seed 42. Both remain almost stationary under forward commands;
  the repaired policy reaches 3 m in 0/8 controlled terrain episodes. Its mean
  GPU utilization was 97.5%. This is not a successful locomotion result.
- AME2 matched 500-update continuations from the same old checkpoint: original
  22/48 goal successes; stumble repair 25/48. Gap, pallets and stones remain
  0/4 each. This small descriptive test does not establish general improvement.
- AME2 map coordinates match the actual gap in eight sampled moving states.
  Holding proprioception fixed and filling holes changes leg targets by
  0.087/0.086/0.119 rad RMS on gap/pallets/stones. The policy receives and uses
  the map; this alone does not establish a correct foot-placement strategy.
- Gap failures include collapse near the edge while a support foot remains on
  the platform. Pallet failures include a foot dropping into a gap before body
  roll increases. These are distinct failure sequences.
- Candidate CPU checks: 50 existing/targeted tests pass. Sixteen actual resets
  remain within configured spawn offsets; all 20 terrain tiles match their
  family columns; all 11,088 valid scan rays hit terrain rather than the robot.

The controlled diagnostic scripts disable randomized play resets and use
structured tiles, so the earlier negative locomotion findings are not explained
away by the public play defects.

## Reference-policy diagnosis and remaining work

The yaw-only 500-update and full-alignment 1001-update G1 runs completed without
forward locomotion. The latter survived all 48 flat episodes but produced almost
zero forward velocity; all 24 terrain episodes stayed near their spawn. Three
reward/input repair rounds therefore did not justify extending that policy.

The community's released `ame1.pt` (iteration metadata 21000) provides an
independent reference. With its original network, control recipe and an explicit
USD-derived joint ordering, it walks in mjlab: commanded speeds 0.25, 0.5 and
1.0 m/s produce approximately 0.281, 0.573 and 1.066 m/s on CPU. Conversion into
the restored native model matches source actor/critic outputs to about 2e-6 and
reproduces the forward behavior. This is **not** a newly trained mjlab policy or
proof of the user's historical training configuration.

It still drifts clockwise and fails pure right turns. The failure side reverses
with a mirrored policy; correcting a 1 mm foot-geometry asymmetry does not improve
it. Restoring source foot shapes changes individual episodes but does not
consistently fix the bias, so those geometry changes are not adopted.

Native videos show real pallet/gap crossings, but stones and beams fail early.
Twenty seconds of constant forward motion can also take a walking policy beyond
the finite terrain; record that separately from obstacle failure. Final terrain
acceptance must include stopping on a supported in-tile target, with true
terminal-state and visual verification.

The retained four-family sparse course is not the community source course.
The source mixes 60% continuous stairs/boxes/rough/slopes with 40% stones/gaps,
uses 8×8 m tiles and larger central platforms. The existing 8×4 m sparse course
was deliberately retained during upgrade, but its learning equivalence was not
established. Diagnose this remaining curriculum difference before another long
G1 run; do not present an approximate terrain substitute as exact source parity.

The radial-beam correction preserves the 8×4 m course, central platform, pit and
14 radial supports. Beam ends meet the rim without protruding into neighboring
tiles. Native geometry checks cover square and both rectangular orientations at
three difficulties, checking every box corner, continuous support along all
beam centerlines, and the retained pit between beams. This fixes a terrain
construction defect; it has not been shown to fix the policy's early falls.
The completed plane screen was frozen at `bbb7f25` and did not use this change.

AME2's fresh 1001-update stumble-only comparison completed on unchanged
`043750d` source. With three matched evaluation seeds, original/repaired goal
successes were 26/20 out of 144 episodes; low-difficulty gap, pallets and stones
each remained 0/6. Both checkpoints represent 24,024 control steps per
environment, only halfway through the 2000-update curriculum. This candidate is
not promoted to formal training.

A separate CPU transfer of the official released Gaze checkpoint reached and
stopped at all four sampled rough/gap goals. Native state replays and terrain
rays confirmed both gap episodes ended on the far platform, not in the pit.
That reference uses a different architecture, 23 actions and a much larger
training sample budget; its success does not validate the native dense learner.
No default algorithm changes follow from the reference test. The behavior and
native replay gates still apply before formal training.


## Restored G1 learning and supported-path diagnostic

The restored community candidate learned actual forward motion in a bounded
1001-update plane run (4096 environments, seed 42). Across evaluation seeds
43/44/45, 47/48 command episodes survived 20 seconds. Commands 0.25 and 0.5 m/s
produced means of 0.248 and 0.494 m/s; moving left/right turns produced about
+0.492/−0.496 rad/s. One 1.0 m/s start fell after 1.34 s, and pure turns retained
about 0.08 m/s forward drift. Plane learning is therefore demonstrated, but full
robustness is not. Network, control recipe and course changed together; this is
not a single-factor attribution experiment.

Native CPU play verified walking, acceleration and stopping over 20 seconds:
10.046 m displacement for 10 m integrated command. The GPU video process had
aborted with exit −6 after training and all three evaluations succeeded; its
original failed state and partial output remain intact. CPU recovery used the
same source and checkpoint, completed with exit 0, and has separate provenance.
Training took 7312.61 s, averaging 13503.8 environment steps/s and 97.76% GPU
utilization, with peak 51279 MiB allocated.

On corrected `2d934e5` sparse geometry, the plane-trained policy reached a
supported in-tile endpoint in 0/8 episodes: seven falls and one backward tile
exit. All four endpoint geometry checks passed. These episodes used supported,
forward-facing starts, so the training reset mismatch below does not explain
away their failure. The policy has not yet learned sparse traversal.

A read-only configuration/geometry audit also found that the narrow +X paths
inherit ±0.5 m XY resets, full-circle yaw and full-circle heading targets from
the broad rough task. At difficulty 0.25, conservative fractions of root
projections over the pit at spawn are 13.95% / 7.875% / 9% for pallets / gaps /
stones. These are geometric bounds, not measured fall rates. Arbitrary heading
commands are incompatible with a one-direction narrow lane.

One artifact-only diagnostic now continues the verified plane checkpoint for
at most 501 updates on all four sparse families, using reset XY/yaw zero,
world-heading target zero and initial level zero; existing level advancement,
positive speed distribution, rewards and model remain. Source is frozen at
`2d934e5`, 4096 environments / 4 minibatches on GPU0. It loads the normal runner
state including optimizer; it is not a fresh matched trial or formal training.
It automatically runs the same three-seed plane checks, unchanged in-tile
endpoint acceptance and native CPU video, then stops for behavior review.
There is no automatic extension or formal-training step. The diagnostic
configuration has not changed production task defaults.


The supported-path diagnostic has now completed all 501 updates and queued
checks. All eight terrain episodes survived their 16–22 s horizons but none
reached the supported endpoint; native video shows standing near the start.
Flat regression fell to 24/48 surviving episodes and only 0.000284 m/s under a
0.5 m/s command. This candidate is rejected, while the earlier plane checkpoint
is preserved. Correcting the start/heading mismatch was insufficient; do not
extend this policy because reward or survival increased. The run took 3761.82 s
at 97.45% mean GPU utilization and 13186.8 environment steps/s.

A bounded CPU official/native dense PPO update comparison also passed: 975
fields, including all compared gradients and optimizer states, had zero error.
It covered timeout bootstrap, GAE, critic-only augmentation, BF16 loss, adaptive
KL and entropy decay using shared native 29-joint symmetry. This rules out a
difference in that tested update, not all possible training/environment bugs.
The sparse-terrain behavior gate remains failed; no further dense training or
formal promotion follows from numerical parity.


## Reward and curriculum follow-up

A matched-initial-state CPU comparison of the retained walking model and the
collapsed model uses identical 0.5 m/s commands and two seed-43 environments per
case. With gamma 0.99, walking/standing discounted returns are 8.6825/5.7131 on
plane but 4.3080/5.7492 on difficulty-0.25 pallets. The walking prior falls after
about 6.21 s and advances only 0.711 m there. The standing model's pallet yaw
term contributes 5.9820 while translation contributes 0.0689. This demonstrates
a preference for standing over the available failed crossing, not over all
possible successful gaits. The exact initial-state hashes match between arms.

Community G1's final base weights match the main current weights, but its base
course retains 60% continuous terrain, and finetuning still retains 20% stairs.
The previous native diagnostic used only sparse paths. Official AME2 instead
conditions heading and standing objectives on reaching a position target; its
coefficients cannot be transplanted directly into this velocity task.

The next bounded contrast changes only the sampling course to 60% continuous
floor and 40% unchanged sparse families, restarting from the retained plane
checkpoint with the same 501-update budget and other settings. It is a minimal
curriculum hypothesis test, not exact reproduction of community terrain. Its
first launch performed no training because another user's task occupied GPU0;
a separate bounded queue waits for GPU0 to become free without interrupting it.
No default task changes or formal training follow before behavior review.


The current shared-resource comparison uses two matched 2048-environment,
16-minibatch, 501-update runs, saving every 25 updates. GPU0 became nearly full
with other workloads, so both AME arms run concurrently on GPU1. Their actual
saved environment/agent configurations differ only in terrain mixing and log
name. Both load the same preserved walking checkpoint and optimizer. Shared
GPU telemetry records AME process memory separately from whole-device memory;
whole-device utilization includes other workloads. The earlier 4-minibatch
control was deliberately cancelled by this task before reconfiguration, and
its logs remain distinct from the earlier unexplained SIGTERM interruption.

## Shared debugging follow-up, 2026-10-19

The user now authorizes bounded diagnostics on all configured tele H100s with
measured memory headroom, while retaining other workloads. Formal training is
explicitly excluded: after both stages pass behavior acceptance, report readiness
and let the user obtain idle GPUs. A briefly idle device does not change this gate.

Both matched 501-update runs completed with finite losses and a sampled peak of
9432 MiB per training process. Their three-seed flat evaluations diverge:
sparse survives only 6/48 episodes (all six are stationary under 1 m/s commands),
while mixed survives 42/48 but does not retain correct command tracking. Mixed
zero-command motion averages 0.250 m/s and -0.452 rad/s; its 0.5 m/s straight
command averages 0.429 m/s and -0.613 rad/s. Thus survival alone hides a large
heading bias. Mixed in-tile terrain acceptance fails all eight episodes through
falls or pit contact. These candidates do not pass the gate.

The mixed run's very short training episodes cannot be equated with deterministic
play performance. Its final action standard deviation is about 0.54, and training
also uses BatchNorm batch statistics. A zero-step frozen-input probe confirms
normal flat spawn/support geometry and sensitivity to batch composition and BN
mode, but does not establish the cause of falls. The next bounded probes compare
matched stochastic rollouts with different BN statistics, intermediate checkpoint
behavior, and 101-update pure-plane continuations using the same 2048-env,
16-minibatch recipe. Experimental scripts stay outside production source.

Those probes have completed. Mixed model 1025 survives 45/48 flat episodes and
sparse model 1100 survives 48/48, but both fail all eight strict in-tile terrain
episodes. Later checkpoint behavior fluctuates substantially, so final-checkpoint
failure should not be described as monotonic forgetting. Neither intermediate
candidate passes the terrain gate or replaces the retained walking checkpoint.

The plane-only controls with training seeds 42/43 survive 46/48 and 44/48 flat
episodes after 101 updates. At 0.5 m/s they achieve 0.501/0.504 m/s with near-zero
mean yaw; failures are confined to 1 m/s tests. This supports short-term retention
of low/mid-speed walking under the 2048-env/16-minibatch recipe, but does not
establish 501-update stability. Both runs finish normally with finite losses.

Frozen-checkpoint train-mode rollouts with batch versus frozen BN statistics
both yield 0/100 survivors at 20 s. Removing Gaussian action noise in the sole
follow-up still yields 0/100 in each arm. Exact initial state/observations and
first-frame policy means match across the two noise conditions. Neither change
alone restored behavior in those probes, so they did not justify a production
BN/noise intervention. They did not test PPO likelihood consistency when rollout
samples are regrouped into optimization minibatches; that distinction matters
for the later audit below.
The base observation configuration already disables noise. The training
distribution includes heading feedback and a range of speeds, unlike fixed-twist
plane acceptance. A boundary audit was corrected from an assumed 8 m width to
the actual 4 m width: 41/58 and 52/58 no-action-noise flat episodes end outside
the original column in batch/frozen BN modes. These are real early terminations
in the mixed course, not evidence that every episode fails on an unbounded
plane. Endpoint location alone does not identify first lost support.

Stage two now derives critic reflection from actual controlled action joints
(`d00f5e0`), preserving the 29-action mapping while correctly handling a 23-action
subset. Existing MDP tests and CPU/GPU single-update checks pass. Official dense
and Gaze teachers use 23 actions and 82 proprioceptive inputs. The diagnostic
native 23-action configuration retains movable PD-controlled wrists and 25
contact channels rather than the official 26, so this is not a complete physical
match. No 23-action learning claim follows from the smoke checks. Both 1024-environment full-terrain memory preflights have completed with
finite updates, and the bounded 1001-update-per-arm learning comparison is now
running from the same clean commit. At the intermediate model_500 checkpoint
(501 updates), both arms fail all 48 fixed-seed first-episode goal screens.
Termination includes both gravity and stagnation; neither arm achieves the
low-speed goal dwell criterion. Saved-state MuJoCo replay confirms loss of
balance on representative gap/stones episodes. Training logs show increased
move-to-goal reward but zero sampled late position/heading rewards and zero
terrain-success EMA at iterations 100/300/500. This identifies the missing
behavioral milestone; it does not isolate its cause or establish final-budget
failure. Both arms have now completed 1001 updates naturally (exit 0), each with
24,600,576 transitions. Final same-seed screening still yields 0/48 goal finishes
and 0/48 low-speed goal holds in both arms. Initial qpos/qvel, goals, headings
and terrain assignments match exactly. The 29-action arm terminates early in
47/48 episodes (mean 4.45 s); the 23-action arm in 44/48 (mean 7.07 s), with
28 stagnation terminations. Both also fail every difficulty-zero episode, so
the high-difficulty evaluation endpoint alone cannot explain the result.
Recorded-state replay of the same gap episode shows both policies remaining
near the start until stagnation at 5.02 s. Reducing the action space alone has
not repaired terrain learning at this budget. This comparison is closed without
extension or a production switch to 23 actions. Its limited sample budget and
remaining physical differences prevent a claim that 23-action learning cannot
succeed, or that the remaining cause has been isolated.


The community base course audit found changes beyond rewards: the native base
uses narrow directional sparse paths, while the community source mixes six
continuous families with stepping stones and concentric gaps in 8 m square
tiles. The declared IsaacLab v2.3.0 source uses a 20-column allocation and
random difficulty within each row. Its grid cell bounds, heightfield sample
counts and spawn heights differ from the corresponding native defaults. These
are confirmed migration differences, not yet a proven sole cause of failure.

An isolated course candidate preserves those distributions and sparse source
surfaces. Sparse validation checks interiors, both sides of vertical cliffs,
finite physical contacts and unfilled gaps; exact cliff-ray hits can differ
because of floating-point side assignment. Source-array native hfields reproduce
the checked signed slopes. Rough native hfields still omit the source steep-edge
XY correction, with measured local discrepancies up to 7.24 cm in the sampled
tile; that limitation remains explicit. The candidate has not been promoted or
shown to learn reliably, and does not change the stable/default course.


The complete 200-tile candidate compiles, and its six sparse collision fixtures
still pass after removal of repeated geometry enumeration. Its 52,478 geoms
also expose a separate scaling limit: MuJoCo Warp 3.11.0 constructs all
upper-triangular geom pairs before filtering, causing a transient observed
~153 GiB host RSS. The 128-env single-update preflight nevertheless exits0
with finite losses; own-process GPU peak is25,226 MiB and wall time1291.8s.
The 2048-env capacity gate exceeded its 42 GiB own-process ceiling during
initialization: sampled peak 46,104 MiB, exit -15 from its own monitor, zero
updates after 1357.7 s. This was neither natural training completion nor a
demonstrated CUDA OOM. The separate 1024-env one-update check subsequently passed
at 42,970 MiB. After GPU1 became exclusively available, the complete 2048-env
course passed its one-update check at 62,478 MiB under a 70 GiB own-process limit
and a 6 GiB device-free guard. All 200 tiles remained present.

The preserved walking policy does not already solve this course. Historical
same-seed screens yielded 3/28 and 5/28 supported-stop successes, all on slopes;
their rectangular stance rule excluded four stones targets. Initial states and
terrain arrays matched, but rollout outcomes varied. The first screen printed
hfield-contact overflow warnings; the repeat recorded none. Their cause remains
unresolved.

The evaluator now places two separate nominal foot footprints using the actual
robot capsule bounds, without requiring support in the empty space between the
feet. This defines all 32 targets. The new gold-policy screen succeeds in 8/32
episodes, all slopes, and falls in the other 24, including all stones and gaps.
No physics-overflow event was recorded. Actual saved-state replay confirms a
representative stones fall at 4.34 s. This is an evaluation correction and a
baseline measurement, not a learned-policy improvement. Experimental scripts
remain outside production source.

A separate 501-update plane continuation from the same preserved checkpoint
completed with 2048 environments and 16 minibatches. It retains 47/48 flat
episodes and tracks 0.25/0.5 m/s commands at approximately 0.258/0.497 m/s, but
achieves only 3/32 supported course stops. The gold, plane and subsequent course
screens share evaluator hashes, goals, standards and exactly equal saved initial
qpos/qvel/origins. Earlier plane controls had run only 101 updates.

An isolated numerical comparison executes the frozen official PPO class and
the native implementation against identical model/storage interfaces and fixed
samples. The 23-action FP32, 23-action BF16 and 29-action BF16 cases match exactly
in rollout quantities, losses and updated actor/critic parameters, including
termination and timeout samples. This does not validate simulator dynamics or
the model port; it provides no justification for changing PPO math at this time.

### Dedicated GPU1 course and likelihood findings (2026-10-20)

The complete-course continuation also finished its fixed 501-update budget with
finite losses. Steady throughput was 8,203 environment steps/s, mean GPU use
94.1%, and own-process memory peaked at 63,044 MiB. This numerical/resource
success did not translate into behavior: the final policy fails all 48 flat
episodes, averaging 1.94 s before termination. The first saved model_1100 already
fails all 16 seed 43 flat episodes. The original walking checkpoint remains
preserved; neither continuation replaces it.

The final course screen records 32 falls, but six trajectories have active
HFIELD overflow and cannot be treated as clean physical evidence. The remaining
26 also fail. The installed collision implementation sets that flag when a
single heightfield pair exceeds MJ_MAXCONPAIR; increasing ordinary nconmax is
not a demonstrated remedy. No overflow warning in a training log proves absence
of overflow. Saved-state replay of two unaffected episodes confirms that the
new policy falls on a gentle slope and the gap's starting platform at about
1.6 s, before complex traversal. Restoring only the gold BN running statistics
to the final policy yields four flat survivors out of 16 but fails command
following, including almost zero motion under a 1 m/s forward command.

A separate audit collects one actual 2048-env, 24-step rollout per plane/course
arm, then recomputes probabilities without any optimizer step. All learned
parameters remain exactly equal, and both rollouts record no physics overflow.
Replaying the original time-step batches gives KL=0 and probability ratio=1.
Regrouping the same samples into 16 minibatches gives mean KL 5.425 on plane and
139.826 on course; approximately 90.94% and 99.992% of original-sample ratios lie
outside the PPO clipping interval. Adding mirrored samples changes these values
very little. This establishes a large pre-optimization batch-composition effect,
not the sufficiency of a BN change for long-term terrain learning.

An isolated candidate therefore fixes gold BN running statistics during both
rollout and update while retaining trainable affine parameters. The first
attempt stops before any optimizer step: ordinary shuffled probabilities match
exactly, but a max-absolute-log-probability check of 0.01 rejects a 0.078 residual
after batch doubling. A frozen-weight audit on saved real terrain inputs finds
the same residual for plain duplication and mirroring, unchanged original
inputs, and exactly equal duplicate/mirror outputs. IEEE FP32 removes it
entirely; TF32 produces mean KL 5.56e-5 with no clipped ratios. This is a numerical
batch-size effect, distinct from the earlier BN effect.

The revised check retains TF32 and all training settings. It requires mean KL
below 1% of the configured adaptive KL target, no original-sample ratio outside
the configured clipping interval, and exact original-input/action and
duplicate-versus-mirror output checks. It preserves the failed attempt and keeps
the original 101-update budget. The revised candidate completes naturally with
finite losses and all 12 actor/critic BN buffer checks unchanged, but fails all
48 flat episodes (mean life 2.52 s) and all 32 course episodes (mean life 3.09 s,
no physics overflow). Saved-state replay shows an early fall near the slope
start and failure on the gap platform. Freezing BN alone is therefore insufficient;
this path is not extended or promoted. Training averages 94.5% GPU utilization
with a 62,788 MiB process-memory peak, which is resource performance, not learning
acceptance.

The community launch script selects a PPO configuration without an explicit
symmetry override, while the native stage-one configuration enables mirrored
data augmentation. A saved-observation forward probe fixes gold weights and BN,
disables TF32, and samples actions from that same policy. Original samples have
KL=0; exact mirrored behavior distributions versus mirrored policy inputs have
mean KL 0.129 on plane and 366.095 on course. About 92.64% of course mirrored
ratios fall outside the PPO clipping interval. This measures lack of equivariance,
not proof of a mirror implementation bug or training causality. A bounded
101-update comparison disables augmentation while retaining every other setting
of the failed fixed-BN control. It completes naturally with finite losses and
unchanged BN buffers, but yields 0/48 full-horizon flat episodes (mean life 2.12 s)
and 0/32 course successes (mean life 2.68 s, no overflow). The course reset states, goals, and standards
exactly match gold. The checkpoint is preserved; this path is not extended.

The full-course training physics invalidates an algorithm-only interpretation
of these comparisons. Existing logs show episode lengths
approaching 1000 control steps, zero terrain advancement, and zero feet-air-time
reward, despite short evaluation falls. Saved first-rollout observations show
course vertical velocity down to -4.59 m/s over the first 0.48 s, with mean
-2.27 m/s; plane mean is -0.12 m/s. This is consistent with free fall, but does not
alone identify a contact defect. A completed no-optimizer probe preserves the
full 200-tile scene and 2048 worlds, with exactly paired qpos/qvel/origins and
identical zero actions. Default SAP produces zero collision candidates and zero
contacts throughout the one-second probe: all robots free-fall at -9.81 m/s by
one second, with median root height 4.10 m below the terrain origin. CPU collision
detection finds contacts at earlier saved states that GPU SAP misses. Overflow
flags remain zero. Changing only broadphase to NXN restores foot contact in all
2048 worlds by 0.2 s, without overflow. This establishes a real collision failure
in the complete training scene, independently of policy or reward.

The installed SAP implementation accumulates candidate work in int32 before
pair filtering. A CPU float32 oracle over the saved scene estimates
260,216,329,690 work packages, wrapping to -1,776,675,366 in int32; the static-only
lower bound already exceeds the limit. This is strong source-backed evidence
for scan overflow, not an exact capture of the GPU scan buffer. The configuration
now explicitly selects the supported NXN broadphase for AME-1. No BN or symmetry
change is promoted. The next short training comparison must use valid contacts,
retain the original reward/algorithm settings, and pass behavior review before
formal training. This physics result does not establish a stage-two fix.

The NXN-only continuation restores real contacts but stops after two updates:
the per-physics-step guard catches HFIELD overflow bit 32 at substep 279 in
environment 680. Its rough tile uses a 161x161 heightfield at 5 cm spacing. A saved
state probe isolates the left thigh capsule and recovers the pre-integration
pose from saved qpos/qvel. That pair reproduces the overflow independently of
learning. This is MuJoCo Warp's internal 50-contact heightfield-pair limit;
raising ordinary nconmax does not address it.

An experimental course builder splits continuous heightfields into two-cell-row
strips, retaining the source height samples and common bottom. The actual G1
collision bodies have bounding radius at most 15 cm, so the narrow strips limit
candidate triangles per pair. Exact touching strips expose numerical ray holes;
a 10-micrometre overlap closes the tested seams. Across nine CPU geometry cases,
the largest measured surface discrepancy is 0.0165 mm. Nine isolated-process Warp
BVH ray cases pass at representative course coordinates. Earlier multi-model
ray diagnostics give inconsistent results and are retained as a limitation,
not used as acceptance evidence. The final builder also passes 500 physics steps
with 2048 full G1 worlds, including recovered failure poses, without any overflow.
These are contact/observation checks, not walking success. A new 101-update
comparison starts from the preserved gold checkpoint with the original reward,
BN and symmetry settings. The strip builder remains experimental pending that
behavior evaluation; the production change remains the explicit NXN selection.

That comparison subsequently completes all 101 updates with finite losses and
no per-substep overflow. Steady GPU utilization is 95.77%, process peak memory
66.85 GiB, and throughput 4,851 environment steps/s. Behavior still fails: only
21/48 flat episodes last 20 seconds, with several forward commands producing
standing, and supported terrain remains 0/32 without overflow. Saved-state
replay confirms a slope fall and standing before a gap. Evaluation initial
qpos/qvel/origins exactly match the gold baseline. Physical validity alone is
therefore insufficient to preserve command following in this continuation.

A zero-optimization diagnostic replaces only the 12 actor/critic BN statistics
and counters with gold values, retaining every learned parameter. On seed 43,
full-horizon flat episodes improve from 9/16 to 13/16; 0.25 m/s tracking recovers
to about 0.207 m/s, with remaining turn/high-speed errors. This is partial evidence
for a BN contribution, not acceptance. A single new 101-update comparison holds
the valid NXN/strip physics fixed and freezes gold BN statistics through rollout
and PPO, while training affine parameters normally. Environment configuration is
identical and the agent configuration differs only in run name; the experimental
code implements the BN intervention. Earlier BN trials used invalid SAP physics
and do not replace this controlled comparison. No production BN change is made.

The stage-two signed-progress 1001-update comparison is also closed: it produces
0/48 goal-and-stop successes, although 9 episodes approach the goal and mean life
increases from 4.45 to 6.35 s relative to the binary reward control. All low-difficulty
gap/pallet/stones cases still fail. Stage-two teacher/critic models have no BN and
disable observation normalization, so the stage-one intervention is not a
stage-two fix. Its own teacher behavior gate remains unmet.

A CPU-only compilation of the unchanged stage-two full 120-tile training scene
checks the installed MuJoCo Warp broadphase selection separately. Its 177,745
filtered pairs select NXN by default, below the 250,000-pair SAP threshold.
Therefore the stage-one SAP scan failure is not an explanation for the stage-two
training failure in this configuration. This check does not step physics or
establish that stage-two contact dynamics and behavior are otherwise correct.
The compiled model and selection report are preserved in
`dedicated_gpu1_20261019/stage2_broadphase_cpu`.

A subsequent GPU check retains the full stage-two training scene and 1024 worlds,
with seed 43, zero actions, ordinary resets and a guard on every physics substep.
It completes 200 physics steps without overflow. At 0.2 s, 998/1024 worlds report foot
contacts, including 51/51 gap, 304/304 stones and 49/51 pallets worlds. This rules out
the stage-one-style global absence of collision candidates in this sampled run.
Zero-action robots subsequently fall and reset; this is not a balance test or a
learned teacher success. Stage-two training failure remains unresolved.

The existing signed-progress curriculum logs also rule against persistent high
terrain levels as the main explanation: the mean level first drops below 0.1 at
update 95, reaches zero by update 400, and ends at 0.00293. Late failures are
dominated by the orientation termination, not stagnation or hip acceleration.
These logged termination values are counts/rates from the logger, not episode
success percentages. Six training logs contain no overflow warning, and the
installed Warp backend enables these warnings by default; this is weaker than
an explicit per-substep check and does not certify every historical rollout.
The evidence does not justify another comparison that only lowers the initial
terrain level, or an unbounded extension of the unsuccessful reward trial.

A paired CPU reference-policy check adds the teacher observation noise and scan
drift at the 1001-update curriculum amplitude, keeping physical parameters and
initial robot/goal/terrain states matched. The published Gaze policy reaches and
holds the goal pose in all four rough/gap cases in both conditions, without early
termination. Strict continuous one-second low-speed stopping falls from 4/4 to
0/4 under noise. This small check supports navigation through the tested noisy
observation path, not noise-free equivalence, native dense-teacher learnability,
or full training-distribution robustness. The existing clean-play acceptance
criterion remains unchanged.

A wider reference-only GPU probe then freezes the teacher schedule at 24,024
control steps and terrain level zero, retaining the full 120-tile training scene,
1024 worlds, all five startup randomizations, training observation noise and
0–1 physics-substep action delay. It completes 3,200 physics substeps without
overflow. The published Gaze policy reaches and holds goal pose in 998/1024 first
episodes: gap 51/51, pallets 51/51 and stones 278/304; the other 26 stone episodes
end early. All noisy episodes fail the strict continuous low-speed criterion.
Saved-state CPU replay shows actual gap/pallet traversal and stone-grid movement
to the goal, plus a stone failure. This strengthens the evidence that these
lowest-level native training conditions permit navigation. It neither proves
full-difficulty robustness nor supplies a learned native dense teacher.

Do not replace the unresolved learning diagnosis with a claim that more updates
must solve it. The historical native teacher already completed 4096 environments
and 5000 updates on source `3b283f4`, before the stumble correction, and still
failed the sampled gaps. The published Gaze checkpoint also has a different
architecture and a much larger recorded training allocation (four 16,000-world
workers). These are not equal-budget or causal architecture comparisons.

### Valid-physics fixed-BN result (2026-10-20)

The isolated fixed-BN arm completes 101 updates without physical overflow; all
12 running-statistic/counter buffers equal the preserved gold checkpoint. It
retains balance in 48/48 flat episodes, compared with 21/48 for the otherwise
identical normal-BN arm. This is partial recovery, not command acceptance:
commands of 0.25/0.5/1.0 m/s yield mean forward speeds of 0.066/0.294/0.608 m/s.
Supported course success remains 0/32 (16 early falls, 16 pre-goal timeouts).
Saved-state replay confirms that the slope case advances but stops 0.32 m from
the goal, while the gap case remains on the starting platform. Evaluation
initial qpos/qvel/origins are checked against the preserved baseline separately.

The checkpoint `f7235bbd92d1a40df8d8dee246881b9611309822e52cf2878951392b8c74e5b1`
and full evidence are backed up. Stable training averages 95.26% GPU utilization
and 4,720.5 environment steps/s; device-memory peak is 68,461 MiB. No BN change
is promoted to production. The next bounded comparison disables only mirrored
PPO augmentation, retaining fixed BN, valid NXN/strip physics and the same
gold initialization and budget. Earlier no-augmentation results used invalid
SAP physics and cannot settle this comparison. No formal training is started.

### Native teacher GPU probability consistency (2026-10-20)

A zero-optimization probe loads the native 29-joint dense checkpoint at update
1000 and collects 24 real training steps from 1024 worlds on the complete
120-tile scene, with the low-level/schedule snapshot used above. All 96 physical
substeps are overflow-free; actor and critic state dictionaries remain unchanged.
The original FP32 batches reproduce their stored probabilities exactly. Shuffled
FP32 batches have no clipping violations. BF16 shuffled batches have mean KL
0.000489 against desired KL 0.01, with 3/24,576 ratios outside [0.8, 1.2]. This
does not support a severe pre-optimization precision mismatch for this checkpoint,
so it does not justify a separate FP32 training run. It is not a gradient audit
or evidence that every checkpoint is numerically equivalent. The real first
minibatch, configuration and report are backed up with matching hashes under
`stage2_gpu_precision/` in the dedicated GPU1 artifacts.

### Valid-physics augmentation control (2026-10-20)

Disabling mirrored PPO augmentation completes 101 finite updates without overflow,
but does not resolve undertracking: flat survival is 46/48 and the 0.25 m/s command
produces only 0.007 m/s. Course success remains 0/32, with 23 early falls versus
16 for the fixed-BN arm with augmentation. The initial states, goals, evaluation
code and criteria match exactly. Replay shows a slope timeout about 1.03 m from
the target and a gap fall at 8.14 s. The checkpoint
`d0d3bdfaa6bc867dbb9cec758b16b6f116677d6ec8e1a5796a6b890260f8c3e5`
is backed up; this arm is not extended or promoted. Its 95.47% GPU utilization
and 5,863.8 environment steps/s are performance results, not behavior acceptance.

The next bounded arm returns to fixed BN **with** augmentation and reloads the
original walking gold model and optimizer. It removes the course script's fixed
spawn/heading overrides, restoring reset XY ±0.5 m, yaw ±3.14 and target heading
±pi. These values are verified in the successful gold run's actual configuration
and agree with the community base distribution. This tests the linked base
spawn/heading distribution, not yaw alone; terrain level zero, joint resets,
material settings, reward and all training budgets remain unchanged. No formal
training or production distribution change is made.


### Base reset distribution control and next teacher contrast (2026-10-20)

Restoring the gold base spawn/heading distribution completes 101 finite updates,
with all 12 frozen BN buffers unchanged. Flat survival falls to 43/48; commanded
0.25 and 0.5 m/s produce 0.075 and 0.341 m/s. The 1 m/s cases include unstable
acceleration and early falls, so their average velocity is not a tracking success.
The unchanged course yields 0/32, with 24 early falls and eight timeouts. Initial
qpos/qvel/origins, goals and evaluation contract match the fixed-BN control.
Replay shows the slope case stopping 0.290 m short and the gap case falling at
8.58 s. Checkpoint SHA256 is
`09d4b50035ca48effa86131e2b4dd6f2bd57a5753ade7766e58fa58c56be8d39`;
the 49,650,306-byte evidence archive and checkpoint are locally hash-verified.
This continuation path is closed without extension or production promotion.
It does not establish that fresh training on the corrected physical course cannot
learn. The original walking gold checkpoint remains preserved.

A saved-state CPU check also bounds foot-slip costs without reconstructing
contacts: fixed-BN/no-augmentation mean all-contact bounds are 0.0337/0.0645
for native site-speed squared, versus 0.0260/0.0403 for body-COM speed norm.
These evaluation snapshots cannot explain training causally or exclude transient
effects; they do not justify another reward-weight intervention.

For the native 23-action dense teacher, the next bounded learning hypothesis is
spawn-yaw curriculum pace at 1024 environments. A draft slowing yaw, goal cone
and map noise together is rejected before execution: five of six existing low
sparse targets would lie outside its final training cone. The replacement only
slows yaw expansion from 48,000 to 192,000 control steps and reuses the original
goal/noise schedule unchanged. CPU configuration/function checks pass; an exact
1024-world, one-update preflight has finite losses and 96 overflow-free physical
steps. The fresh 1001-update contrast has now completed with source d00f5e0,
seed43, the existing network/reward/optimizer and the original 48-case evaluation.
All 96,096 physical steps are overflow-free. Matched initial qpos/qvel, goals,
headings and terrain assignments are exactly equal to the baseline. Low-speed
goal holding remains 0/48. Cases ever within 0.5 m increase from 1 to 10, but
gravity terminations increase from 15 to 33; all six low gap/pallets/stones
cases fail. Mean lifetime changes from 7.073 to 6.798 s. This changes movement
tendency without establishing reliable terrain traversal or goal stopping.
Checkpoint SHA256 is
`7fea851dac374bdc4d66910389e8811128a34a79fabfcb2cc764af5fe8d724a9`.
The arm is not extended or promoted to a production curriculum. Formal training
remains gated on the full behavior requirements.

The next stage-one check is capacity only: fresh weights, the corrected complete
course, native base reset distribution, normal BN, 2048 environments and four
minibatches for one update. Previous valid course continuations used 16
minibatches and only 4.96 million new transitions, while the successful fresh
plane run used four minibatches and 98.40 million transitions. The successful
16-minibatch plane continuation prevents attributing course failure to minibatch
count alone. A capacity check cannot establish learning success or authorize
formal training.


The fresh 2048-environment/four-minibatch capacity check completes one finite
update and 96 overflow-free physical steps. Sampled process peak is 67,818 MiB;
wall time is 1099.65 s, dominated by scene initialization. Its checkpoint is
preserved but is not used to initialize the learning experiment.

The initial 2048-environment learning attempt is stopped by the existing memory
monitor before completing its first update (SIGTERM, zero logged updates).
Both attempts reach 63,356 MiB process memory during rollout; the longer attempt
jumps to a sampled 75,432 MiB at the first optimizer phase, with only 3696 MiB
free. Saved runtime configurations match except for budget, save cadence and
run name. This demonstrates that the one-update capacity result underestimated
the subsequent startup peak; the exact allocator/workspace cause is unproven.
Asynchronous device/process samples do not establish an external competing job.

The revised bounded diagnostic lowers the environment count to 1792 and keeps
four minibatches, the original backend, reward, augmentation and physical
helpers. Its 2288 updates give exactly 98,402,304 transitions, matching the gold
plane sample count but using 45,760 optimizer steps instead of 20,020. This is a
full-course learnability test, not an isolated minibatch causal claim. The
70 GiB process/6 GiB free-memory guards remain. Allocator peaks are recorded
alongside finite losses; no automatic capacity fallback or extension is added.
The original flat48/course32 evaluations follow successful completion, and
formal training still requires actual behavior acceptance.


The 1792-environment attempt also stops before update one: sampled process
peak 76,956 MiB, device free 4126 MiB. No behavioral conclusion follows.
An isolated diagnostic using the actual shared CNN, saved own weights and
its augmented minibatch shape measures PyTorch peak allocation of 22,813 MiB
with cuDNN benchmark enabled versus 9045 MiB disabled. External sampling misses
these transient peaks. This supports testing autotuning workspace as a cause,
not a complete PPO/physics capacity guarantee. A five-update full-scene check
now disables only that backend flag in the artifact entry script, after the
runner backend setup. The physical and reward configuration, 1792 environments,
four minibatches and memory guards remain unchanged. No production backend
change or automatic subsequent learning run has been made.


The five-update full-scene check with cuDNN benchmark disabled still stops
before update one at 76,760 MiB process memory. The isolated CNN result therefore
does not establish a sufficient fix. A subsequent actual PPO.update diagnostic
with synthetic observations, own saved weights and the real shared model peaks
at 19,380 MiB reserved memory for four minibatches. Suppressing gradients only
for the logging-only mirror loss does not reduce that reserved peak and does
not produce exact final parameter/BN parity; that candidate is not adopted.
Eight minibatches with the original gradient path reduce the measured reserved
peak to 10,072 MiB and complete forty optimizer steps. This supports a new
five-update full-scene capacity check at 1792 environments, not a behavior claim.
The physical/reward configuration is unchanged and existing guards remain.
Eight minibatches double optimizer steps per rollout compared with four, so any
subsequent learning comparison must record that protocol difference explicitly.


The eight-minibatch full-scene capacity check completes naturally: five finite
updates, 480 overflow-free physical steps, no memory guard, and a 67,452 MiB
own-process peak. Internal PyTorch reserved memory peaks at 10,100 MiB.
Checkpoint SHA256 is
`0f3de210a138f932eb064bb652137fe764a5bb1b553a0caf3dcc0ba33b7194a6`.
The short sample measures about 3619 transitions/s, but an unrelated playback
process overlaps and device utilization cannot be attributed solely to AME.
Actual runtime environment configuration matches the failed four-minibatch
control. Prepared-versus-runtime terrain size/count differences are the verified
Scene/TerrainGenerator initialization normalization, not hidden reward changes.
After the authorized playback release, the bounded fresh-course diagnostic
started on October 20. Capacity success does not establish terrain behavior.

## Completed fresh-course diagnostic and matched plane control

The completed run used frozen source `2d934e5`, fresh seed 42, 1792 environments,
eight minibatches and five learning epochs per 24-step rollout. Its fixed
2288-update budget is 98,402,304 transitions, equal to the retained gold plane
run's sample count. It performs 91,520 optimizer steps versus gold's 20,020:
this is not an otherwise matched terrain comparison. Rewards, base resets,
normal BatchNorm training and augmentation are unchanged. The complete
200-tile course uses NXN and the experimental heightfield strips.

After the initial unrelated playback ended, recent 50-update windows measured
about 4790–4800 transitions/s, roughly 96% device utilization and 67,452 MiB
own-process memory. These are observed diagnostic results, not a promised
formal-training capacity. Existing memory, finite-loss and per-physics-step
overflow guards remain active.

CPU flat diagnostics at checkpoints 1250 and 2000 used the same frozen source,
seed 43 and 16 fixed-command episodes as the retained gold control. All three
survived 20 seconds in every episode; their command responses differ:

| Forward command (m/s) | Checkpoint 1250 | Checkpoint 2000 | Gold |
| --- | ---: | ---: | ---: |
| 0.25 | 0.156 | 0.248 | 0.248 |
| 0.5 | approximately 0 | 0.322 | 0.493 |
| 1.0 | approximately 0 | approximately 0 | 0.994 |
| Stand | 0.163 | 0.073 | approximately 0 |

Entries are measured forward velocities, not reward scores. Low-speed and
combined forward/turn behavior improved, while high-speed following and
stationary stopping remain unresolved. A few slope, inverse-slope, grid and
rough curriculum columns began advancing after checkpoint 1750; this does not
establish terrain traversal. Training-log twist errors accumulate over an
episode and are normalized by configured command duration, so they cannot be
read as per-second mean tracking errors or compared without considering episode
length. The fixed-command behavior measurements use active-step means.

The run completed naturally with exit code 0, 2288 finite updates and 219,648
overflow-free physics steps. Total wall time was 21,760 s, with steady throughput
4777 transitions/s. The complete training archive and final checkpoint were
downloaded and independently hash-verified; final checkpoint SHA256 is
`c5d5c23ad71b74fa1281ff62d919e024d90bfa7cd8230dbc502b5747f7953e19`.

This is not the full community training budget. Its registered AME task defaults
to 4096 environments, 10,000 updates and 24 steps per update: 983,040,000
transitions. This diagnostic supplies 10.01% of those samples, 22.88% of the
per-environment control steps and 45.76% of the optimizer steps because it uses
eight rather than four minibatches. These defaults do not establish the released
checkpoint's actual training budget. Failure here establishes a failed bounded
screen, not failure after full reference training; more training remains a
hypothesis to assess from learning progress, not an automatic remedy.

Final serial GPU evaluation survives all 48 flat cases, but commanded forward
speeds 0.25/0.5/1.0 produce only 0.179/0.327/0.313 m/s. Stand-command drift falls
to 0.0038 m/s. All six 1.0 m/s cases survive and produce only 0.309–0.318 m/s;
the deficit is consistent across these three evaluation seeds, which do not
constitute independent training seeds. The policy has learned some movement
and stopping; reliable speed following remains unsolved. On the unchanged
32-case terrain contract it reaches
and holds 3 targets: one grid and two low-difficulty inverse slopes. Stairs reach
the target region but overshoot, while stones/gaps never reach it. Initial
states match the existing baseline exactly; all targets are defined and active
trajectories are overflow-free. Saved-state CPU panels confirm the staircase
traversal and sparse-terrain stagnation; support/hold conclusions use original
GPU measurements rather than reconstructed CPU contact forces.

A paired CPU test also exposes delayed stopping on flat ground. With identical
seed-43 initial states, the original 16 commands run for 10 seconds, then switch
to zero for 10 seconds. After the 0.5 m/s command, maximum subsequent displacement
is 0.628 m for the course policy versus 0.205 m for gold; completing a continuous
two-second low-speed hold takes 5.72–5.86 s versus 2.62–2.64 s. This is a dynamic
response problem even without terrain, not evidence of a unique training cause.
Gold also falls in one right-yaw stopping case, so it is not a fully accepted
reference. These CPU hold measurements check speed and yaw rate only; they do
not replace the terrain support and posture criteria.

A separate CPU course comparison uses the converted released community policy
and this final course policy, with identical initial qpos/qvel/origins, goals,
source and evaluation standards. Both complete naturally without active-trajectory
overflow. Strict success is 0/32 for the reference and 2/32 for our policy, but
that score alone reverses important aspects of their behavior. In the four gap
cases, mean maximum forward progress is 3.30 m versus 0.77 m. The reference
crosses gaps in the reviewed replay but misses the target laterally; our policy
stops near the starting platform. In all eight stones/gaps cases, our policy
still receives 0.5 m/s throughout the final five seconds while measured mean
planar speed is only 0.00035–0.00199 m/s. The evaluator has not commanded this
standstill. Other failures include overshooting stairs and remaining unsettled
on slopes. Preserve the strict target criterion while reporting progress,
directional drift and stopping separately. This CPU comparison identifies
different failure modes; it neither proves a unique training cause nor replaces
GPU acceptance, and the policies have different training histories.

The fresh plane control, started after the final GPU behavior review, uses
the same environment count, optimizer protocol, seed and sample budget. It
replaces terrain with a plane and disables terrain curriculum, retaining the
other settings. This distinguishes terrain effects from the changed
1792-environment/eight-minibatch protocol; it is not an extension or a formal
training run. Its final flat48/course32 evaluation remains serial.

The repair branch now integrates the community course and heightfield strips
under `terrains/community`, selected by the existing G1 base train/play task.
The measured capacity defaults are 1792 environments, eight minibatches, NXN
and disabled cuDNN autotuning; no alternate launcher or fallback is introduced.
Paper, G1 finetune and AME-2 retain their existing terrain selections. Fifty
existing CI contract tests report success across configuration, observation,
CLI, checkpoint and AME-2 interfaces. Sixteen family/difficulty cases
match the frozen reference in compiled geometry arrays, origins and 1008 native
ray queries. The public CPU training entry point completes one actual PPO update with two
environments on the full course: 40 optimizer steps, finite losses and all
156 checkpoint tensors finite. The saved model is locally hash-verified.
Public CPU play also exits successfully with eight recorded steps: all 48
recorded tensors are finite, attention weights normalize correctly, and the
eight position records use independent storage. The play dump is independently
hash-verified locally. These checks establish the train/save/load/play path,
not policy behavior. The existing one-environment symmetry smoke fixture now
collects eight steps so each of the eight minibatches receives a sample;
its targeted full-scene CPU regression passes with a natural exit code of zero
(1104 seconds including scene initialization).
Actual environment YAML matches the original course run except for the intended
environment count and its propagated terrain count. This integration does not establish behavioral
acceptance or a completed migration. Stage two still requires the project's
own teacher to reach and stop reliably on sparse terrain; the completed yaw
schedule contrast above did not meet that requirement.


The matched fresh-plane control completed naturally: 2288 finite updates,
98,402,304 transitions and 219,648 overflow-free physics steps. Its complete
archive and final checkpoint were independently verified locally (checkpoint
`6924eccdc61b468c4e833372c14a31d204dbf78d3554aed71cdd87e386ae0412`).
Steady throughput was 11,611 transitions/s, sampled own memory 12,518 MiB,
and wall time 8506 s. Flat survival is 44/48. Commands 0.25 and 0.5 m/s yield
0.276 and 0.532 m/s in all six respective cases; four of six 1.0 m/s cases
fall within 1.42–1.52 seconds, while the two survivors track 1.005 m/s.
Do not average falling accelerations into a claim of high-speed tracking.
Pure-yaw response remains low and stand-command drift is 0.0266 m/s.
The identical optimizer protocol can learn low/mid-speed locomotion, but
this policy also fails robust flat acceptance.

The unchanged terrain test passes 8/32, all slopes/inverse slopes. Initial
qpos/qvel/origins and the evaluation contract match the prior GPU baseline
exactly. All eight family replay panels were inspected: stairs stagnate,
inverse stairs/grid/rough fall early, and sparse terrain is not solved.
This narrows the course-policy deficit to terrain-conditioned acquisition
without proving a unique reward defect.

The subsequent bounded continuation stopped after 1974 of 2288 updates because
the physical guard detected `EPA_HORIZON` overflow in a heightfield/foot-capsule
collision. It did not finish naturally. The retained checkpoint 4037 passes
14/32 original terrain cases, including all four `stairs` cases. These cases
move outward from a raised central platform and therefore test **descent**;
they do not establish ascent. Inverse stairs, rough terrain, stones and gaps
remain unsuccessful. Initial states and goals match the earlier evaluation;
the compiled models differ only in their name-lookup array among 481 arrays.

The saved collision reproduces the capacity-24 overflow on both CPU and GPU.
Increasing the process-local EPA horizon capacity to 64 clears it without
changing the final contact rows in that forward-only replay. This explains the
interruption, not the locomotion deficit. The bounded recovery from checkpoint
4037 finishes all 537 updates naturally, with finite losses and 51,552
overflow-free physics steps. Steady throughput is 4797 transitions/s, GPU
utilization 96.4%, and peak process memory 65.9 GiB. Its endpoint restores the
original 4576 effective-update budget; 223 updates after the saved checkpoint
were discarded. Simulator state and terrain levels restart, so this is not an
uninterrupted continuation.

The recovered checkpoint survives all 48 flat cases but remains stationary in
all six direct 1 m/s cases. Mean speed under the 0.5 m/s command is 0.493 m/s.
The original terrain evaluation passes 13/32 cases: descent 4/4, both slope
families 8/8, and grid 1/4. Ascent, rough terrain, stones and gaps remain
unsuccessful. All four evaluation runs complete without physical overflow.
The capacity correction therefore does not establish behavior acceptance.

G1 base train, play and evaluation entry points now call the same scoped
capacity helper before environment construction. It sets the three imported
private mujoco-warp 3.11.0 constants in this process only; dependency files and
other task configurations are unchanged. The actual helper clears the saved
failure on CPU with identical contact output to the diagnostic replay; its GPU
replay also completes without overflow. Actual public CPU entry checks on the
full 200-tile course complete naturally: train performs one PPO update and 96
physics steps, play records eight steps (32 physics steps), and eval completes
one episode (276 physics steps). All are overflow-free, with capacity 64
asserted before environment construction. The training checkpoint and all play
tensors match the earlier capacity-24 CPU check exactly. The fresh one-update
checkpoint fails its gap episode; these checks validate execution, not learned
behavior. Public-entry GPU runtime verification remains pending; the complete
recovery above used the equivalent diagnostic setting.

On flat ground, checkpoint 4037 remains stationary when given 1 m/s directly,
but reaches about 0.632 m/s after five seconds at 0.5 m/s. A paired CPU reward
audit reproduces the recorded trajectories exactly. With the common 1 m/s
target after six seconds, weighted reward rates are 2.824 for standing and
2.841 for the available walking gait. This small advantage supports investigating
the reward landscape, but does not establish that an accurate 1 m/s gait would
score poorly. A matched single-variable contrast widens only the linear
tracking kernel from 0.25 to 0.5. At 251 continuation updates, the original CPU
flat evaluation (seed 43) reports mean speed 0.675 m/s under the 1 m/s command,
compared with approximately zero in the narrow-kernel baseline. Both survive
16/16 cases, but mean speed under the 0.5 m/s command drops from 0.482 to
0.430 m/s. This supports a reward-related learning limitation while exposing
a tracking tradeoff. The completed 537-update comparison survives all 48 flat
cases and reaches mean speeds 0.179/0.433/0.746 m/s for commands
0.25/0.5/1.0 m/s. The narrow-kernel baseline reaches 0.206/0.493/approximately
zero. Thus direct high-speed initiation improves, while low-speed tracking
regresses. Original terrain success increases from 13/32 to 15/32: descent
4/4, slopes 8/8, grid 2/4 and rough terrain 1/4; ascent, stones and gaps remain
0/4 each. All evaluation physics steps are overflow-free.

Paired saved-state replay confirms the added grid and rough-terrain goal holds.
Both policies still stop near the first ascending steps. On the inspected gap,
the baseline falls near the first gap while the wider-kernel policy waits there
until timeout. Another grid case changes from timeout to a fall. Initial states
and goals match, and compiled models differ only in their name-lookup array.
This is partial progress on a single training seed, not reliable terrain
learning. Preserve both checkpoints; the default reward remains unchanged.

The AME-2 fresh flat-surface contrast completes 1001 updates with 1024
environments and no physical overflow. It improves proximity and survival but
still achieves 0/48 goal-pose holds; its native-terrain evaluation also achieves
0/48. All six low-difficulty gap/pallet/stone cases lose balance within
0.62–1.22 seconds. Flat locomotion does not establish terrain transfer. This
24.6-million-transition diagnostic uses approximately 5% of the local official
dense teacher configuration's default 5000 × 4096 × 24 sampling budget; it is
not an equivalent full-training comparison. A native-terrain 4096-environment
capacity check completes five finite updates and 480 overflow-free physics
steps. Neither capacity nor reward improvement
can replace the required behavior validation before formal training.


A bounded native-terrain AME-2 diagnostic restores the official batch size of
4096 environments for 1001 updates, with the same rewards and curriculum as
the earlier 1024-environment run. It completes naturally with 98,402,304
transitions, finite losses and 96,096 overflow-free physics steps. Steady
throughput is 12,360 transitions/s, sampled GPU utilization after the first
300 seconds averages 94.4%, and peak process memory is 45.1 GiB. This is four
times the experience at the same update count, approximately 20% of the local
official default sampling budget; batch and sample effects remain inseparable.

The original flat evaluation reaches and holds the target pose in 48/48 cases;
41/48 also satisfy the continuous one-second low-speed pose criterion. Native
terrain evaluations with seeds 43 and 44 satisfy that criterion in 8/48 and
6/48 cases. The retained 1024-environment endpoint has zero such holds in the
matched seed-43 flat and terrain evaluations. Initial states, goals, compiled
models and evaluation protocols match. These are evaluation seeds of one
trained policy, not independent training replications.

Saved-trajectory replay shows the new policy reaching and settling near rough
and boxes targets, while the inspected low gap still causes stagnation at its
edge. Gap, pallets and stones remain unsuccessful in both terrain evaluations.
Low-speed pose screening and saved-state pictures do not alone establish foot
support. Preserve this improved navigation checkpoint, but do not treat it as
a reliable terrain teacher or proceed to formal training on this evidence.


A matched 251-update continuation from the same native-4096 checkpoint tests
one reward change: rotating torso COM velocity into the goal's yaw frame
instead of the full body frame. Both arms complete 24,674,304 transitions and
24,096 overflow-free physics steps. Training initial arrays and all three
original evaluation initial arrays match; evaluation model hashes match too.
This formula is also present in the pinned official source, so the contrast
investigates a reward-design hypothesis rather than a proven migration bug.

The control achieves low-speed goal-pose holds in 46/48 flat cases and 9/48
terrain cases for each of evaluation seeds 43 and 44. The yaw-frame candidate
achieves 30/48, 8/48 and 9/48 respectively. Gap, pallets and stones have zero
arrivals in both arms and both terrain seeds. Saved-state replay shows the
candidate waiting at the gap edge, falling on pallets at 3.42 seconds and on
stones at 2.60 seconds. More ordinary-terrain proximity does not compensate
for the stopping regression or establish sparse traversal. Close this
frame-only hypothesis at the fixed budget; preserve both endpoints and keep
the default reward unchanged. These remain two evaluation seeds of one
training seed, not independent training replications.

A separate H100 80GB capacity comparison completes 31 finite updates at both
4096 and 6144 environments, with 2976 overflow-free physics steps per run.
Excluding the first ten updates, throughput is 12,204 versus 12,934 environment
transitions/s: about 6.0% more throughput for 50% more environments. Peak
process memory is 45.2 versus 66.4 GiB; the larger run retains about 12.8 GiB
of device headroom. This supports 6144 as a measured option for new bounded
diagnostics, not a default or behavior promotion. Changing environment count
also changes the PPO batch and experience per update; preserve frozen paired
comparisons rather than changing their size midway.

### 学会徘徊后的阶段二配对回报核查（2026-10-21）

将当前 control1250 策略与官方 Gaze 策略放在相同原生物理、初始状态、目标及随机种子下，CPU 执行两个 rough 和两个最低难度 gap 案例，共 6,400 个无溢出物理步。官方策略四例均达到姿态停稳；本地策略两例 rough 停稳，两例 gap 在 10.04 秒因停滞结束。两例 gap 的完整折扣回报分别为官方 10.315/9.003、本地 7.440/7.973。因此，这组行为不支持“当前总奖励更偏好沟边徘徊”的解释；单个进度奖励被利用仍不能解释全部学习失败。该比较使用不同架构与训练预算的策略，不能据此证明迁移无误或定位唯一原因。

诊断中发现推理加载入口只加载 actor，首次读取的 critic 数值无效，已明确作废。显式加载并逐张量校验 checkpoint critic 后复核，初始状态、逐步奖励及行为结果均与首次一致；未修改训练实现。证据：`stage2_control1250_reward_audit_v2/return_review.json`、`critic_correction_review.json`。

同日 dewei 的两个 AME 实验及其控制进程异常消失，没有正常结束或显存保护记录；随后 GPU5–7 出现外部三卡训练。退出原因尚未确定，不能认定人为终止或 OOM。阶段一 gamma 对照仅完成 7 次更新，不能作为行为结论；公开 GPU 入口验证尚未完成。按资源协调决定，固定阶段一对照已迁回原 GPU1，从原始 checkpoint 重新开始既定 251 次更新，完成后串行执行原行为评估。中断现场和诊断完整归档已独立校验；dewei 等待重新协调。


### gamma 对照及阶段二训练域采样（2026-10-21）

原 GPU1 上的固定 gamma.995 对照已自然完成 251 次有限更新（10,795,008 transitions、24,096 个无溢出物理步）。稳态吞吐约 4,777 transitions/s，平均 GPU 利用率 96.27%，进程显存峰值 65.87 GiB。原协议三个平地评估种子共 48/48 存活 20 秒，但直接 1 m/s 指令的六个案例均速约 −0.00079 m/s；地形通过数为 12/32，低于 gamma.99 对照的 14/32。配对评估初态与几何相同；五组保存轨迹回放可见上楼和跨沟仍受阻，另有 grid 新摔倒。仅延长回报时间范围没有修复高速启动或可靠地形行为，关闭该路径并保留两组 checkpoint。证据：`course_credit_gamma995_251_final_comparison.json`、`course_credit_gamma995_251_visual_review.json`。

随后用阶段二 control1250 在原训练域完成一次固定随机策略采样：4,096 环境、68×24 步、6,684,672 transitions；6,528 个物理步无溢出，所有可训练参数逐位不变，未执行优化器更新。首次不完整回合单独统计。完整回合中 gap 491、pallets 488、stones 3,720 次均无终点距离小于 0.5 m 的样本，原课程成功计数也均为零；rough 则 220 次中有 217 次课程成功。这定位到当前稀疏地形成功探索不足，而不是所有地形都无法获得到达反馈。优势与朝向目标进度的相关统计不构成梯度因果解释，也不能替代停稳与支撑验收。证据：`stage2_onpolicy_credit_probe/review.json` 及原始 episodes/rollouts。

探针自然完成后，已在同一 GPU1 串行启动固定公开 GPU 入口验证（train5→save/reload→play100→eval64）；当前结果待定，不算行为通过。dewei 继续等待用户协调，不扩训练预算，不进入正式训练。
