# Stochastic Two-Room Evaluation

This experiment evaluates the existing Two-Room checkpoint without retraining.
It measures transfer under changes to dynamics and geometry, not whether the
checkpoint has learned the new stochastic transition distribution.

## Settings

The base environment, renderer, speed (5 pixels/step), action bounds and success
threshold (distance < 16 pixels) come from `stable_worldmodel`'s TwoRoom-v1.
The new environment subclasses it in `stochastic_tworoom.py`.

| Scenario | Visible doors | Hidden closure | Transition noise | Sliding region |
| --- | --- | --- | --- | --- |
| original_geometry | y=49 | none | none | none |
| two_doors_open | y=49,175 | none | none | none |
| hidden_door | y=49,175 | one, uniformly sampled at reset | none | none |
| position_noise | y=49,175 | none | Gaussian, std=2 pixels/axis/step | none |
| sliding | y=49,175 | none | none | action displacement gain 0.5 or 1.5 |
| combined | y=49,175 | one, uniformly sampled at reset | std=2 pixels/axis/step | gain 0.5 or 1.5 |

Door half-widths remain 14 pixels. The hidden closure is fixed for the episode,
affects collisions only, and never changes rendering, proprioception, goal
images or policy info keys. It can be inferred from interaction. For an action
sequence that reaches one door, the same initial observation admits pass and
block outcomes. Two modes are not guaranteed for arbitrary action sequences.

Position noise changes the physical position, not image pixels or sensor values.
It is sampled independently per axis per physical step before collision
resolution. Images and proprioception report the resulting physical position.
Sliding applies when the pre-step center is inside x=[28,98], y=[28,196].
Each physical step in that region independently samples a scalar gain from
{0.5,1.5} with equal probability, shared across both action axes. The region and
gain are not rendered. Transition noise and sliding gains use independent RNGs.
Actions are clipped to [-1,1] before gains are applied, so amplification really
increases displacement. All interventions retain the base collision rules.

## Protocol

The suite defaults to 50 matched start/goal pairs, seed 42, the epoch-20
checkpoint, 50 physical steps, CEM with 300 samples and 30 iterations, planning
horizon 5, action block 5, and replanning every 25 physical steps. These planner
settings are inherited from the existing evaluation, including its infrequent
replanning; the agent cannot immediately react after discovering a blocked door.

Only dataset pairs with endpoints in opposite rooms and outside the central
wall's collision band are eligible. All scenarios use identical dataset pairs
and environment seeds. Current and goal images are rendered from the simulator,
rather than copied from the original one-door dataset. The original geometry
scenario uses this new matched cohort, so its result should not be equated with
the previously recorded 48/50 on the original unfiltered cohort.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p test_stochastic_tworoom.py -v
.venv/bin/python eval_stochastic_tworoom.py
```

The suite sets storage and CPU thread environment variables, evaluates scenarios
sequentially, and refuses to overwrite existing metrics. Use a fresh output
directory for a new run:

```bash
.venv/bin/python eval_stochastic_tworoom.py \
  --seed 43 --output logs/eval_results/tworoom_stochastic_seed43
```

For a single configurable run:

```bash
STABLEWM_HOME="$PWD/stable-wm" OMP_NUM_THREADS=1 .venv/bin/python eval.py \
  --config-name=tworoom_stochastic \
  policy="$PWD/stable-wm/checkpoints/flow-jepa-tworoom-paper-4gpu/weights_epoch_20.pt" \
  world.position_noise_std=2.0 world.sliding=true \
  output.dir=logs/eval_results/tworoom_custom
```

`world.forced_closed_door=0` or `=1` forces a branch for diagnostics; null samples
uniformly. `stochastic.environment_seed` controls environment randomness
independently of dataset selection, CEM and flow seeds. Setting
`stochastic.cross_room_only=false` restores unrestricted task sampling.

Each scenario saves `metrics.json` with resolved configuration and per-episode
results, `trajectories.npz` with physical positions/actions/applied gains, an
initial observation PNG, and a log. Hidden state appears in saved diagnostics
only, never policy inputs. Suite summaries contain Wilson 95% intervals,
differences against both-doors-open, and the checkpoint SHA-256. Fifty trials at
one seed are a pilot; the intervals do not capture training-seed uncertainty.
Trajectory arrays use (time, episode, coordinate). Completed episodes freeze;
their unused subsequent action slots and gain slots may contain NaN. Only the
first `executed_steps` actions for each episode were sent to the simulator.

The first completed experiment is documented in
[the October 20 results](stochastic_tworoom_results_20261020.md).

## Interpretation

The two-open control separates visible geometry shift from hidden dynamics.
`blocked_episodes` counts episodes where the hidden closure changed a transition;
`sliding_episodes` counts episodes that entered the sliding region. Success
also depends on whether the step budget permits the detour through the open
door. `optimistic_min_steps` lower-bounds necessary travel from the maximum
per-axis speed and vertical displacement via an available door; exceeding the
budget proves infeasibility under the base collision rules. Smaller bounds do
not prove reachability. It allows the 16-pixel goal tolerance and uses the largest sliding gain everywhere
when sliding is enabled, and is omitted for unbounded Gaussian position noise.
`budget_infeasible_lower_bound_episodes` counts tasks excluded by this bound.
Success rates alone cannot show that Flow-JEPA predicts both branches or calibrated
probabilities. That would require stochastic training data and repeated latent
predictions compared to repeated simulator futures for fixed state/action
sequences, with a deterministic world-model baseline.
