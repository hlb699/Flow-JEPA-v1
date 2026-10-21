# Stochastic PushT Evaluation

Completed pilot: [2026-10-20 results](stochastic_pusht_results_20261020.md).

This experiment evaluates an existing PushT checkpoint without training or
modifying the original HDF5 dataset. Dataset states and future states define
matched start/goal tasks; changes act on the online Pymunk simulator.

## Dynamics

`stochastic_pusht.py` subclasses the installed `PushT` environment. It preserves
the shape geometry, relative actions, PD controller, wall collisions, original
space damping (0), rendering, and success rule. The simulation uses 10 control
steps/second with 10 physics substeps of 0.01 seconds per control step.

| Intervention | Distribution | Resampling |
| --- | --- | --- |
| Mass | Base mass times Uniform(0.5, 2.0) | Once per episode |
| Contact friction | Uniform(0.1, 1.0) | Once per episode |
| External force | Independent Normal(0, 2000^2) components in world x/y | Every control step |
| Contact turn | Angular-velocity impulse of -5 or +5 rad/s, equal probability | Every control step with agent/block contact |

Mass and rotational inertia are scaled by the same factor, preserving shape
geometry. Mass alone can have little effect in this kinematic-pusher setup:
there is one dynamic object, no gravity, and no inertia changes relative to mass.
With external force enabled, the same force produces acceleration proportional
to 1/mass. The regression test verifies this relationship in the simulator.

Friction means agent/block **contact friction**, not block/table friction. The
base simulator has no explicit table-contact model. Importantly, the original
code assigns `Body.friction=1`, while Chipmunk uses `Shape.friction`; the original
agent and block shapes actually have coefficient 0. The intervention assigns
agent shape coefficient 1 and both T shape coefficients the sampled value.
Chipmunk multiplies the two shape coefficients, giving the intended contact
coefficient. Wall friction remains unchanged. A fixed-coefficient control uses
0.55, the mean of the random-friction distribution, to distinguish increased
friction from its variability.

Force acts on the T block's center of mass, not the pusher. A single world-space
force is held for all 10 substeps of a control step. The body velocity callback
adds F/m to acceleration on every substep, equivalent to continually reapplying
the force that Pymunk would otherwise clear after each substep. Forces use
simulation units, not physical SI units. With original mass 1 and damping 0,
the scale corresponds to roughly 2 pixels of free displacement per control step
per axis, with Gaussian tails.

Contact turn acts on T angular velocity, leaving angle and position continuous.
Only agent/T contacts trigger it; T/wall and agent/wall contacts do not. At most
one impulse is injected per control step, even if both parts of the T report
multiple contacts. Continuous contact can therefore trigger again on the next
control step. The base damping 0 quickly removes the added angular velocity,
but position integration creates a rotation of roughly 0.05 rad (2.9 degrees)
per impulse when no other collisions intervene. An impulse on the last substep
can affect the angle on the next control step.

All four interventions use separate seeded RNG streams. Parameters, forces and
impulses do not enter policy observations or rendering. They are saved only as
diagnostics. Parameters are applied after the inherited reset settling steps;
interventions are disabled during reset so matched tasks have identical initial
physical states and images.

## Scenarios

The suite runs clean, fixed-friction reference, mass only, friction only,
mass/friction together, force only, contact turn only, and all interventions.
The default checkpoint is the existing epoch-20
`flow-jepa-pusht-paper-1gpu-strict`, whose historical clean result is 44/50.

The default protocol uses 50 paired tasks, seed 42, goal offset 25, budget 50,
CEM with 300 candidates and 10 iterations, 5 prediction steps, action block 5,
and replanning every 25 physical steps. Dataset row sampling retains the legacy
evaluator's selection convention to match its cohort. The new evaluator supplies
dataset state and goal via reset options and uses actual simulator images. This
avoids showing an image copied from the dataset while the physical state has
already moved during reset. The new clean control is the comparison reference;
any difference from the historical result can also reflect initialization.

Success retains the original environment rule: the Euclidean error of the
combined agent/block positions is <20 pixels and wrapped block-angle error is
<pi/9. This is not an object-coverage-only metric. The budget is unchanged and
perturbations add no reward changes or video overlays.

## Running

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_stochastic_*.py' -v
.venv/bin/python eval_stochastic_pusht.py --gpus 0 1 2 3
```

Each GPU runs one evaluation subprocess at a time. A single GPU is the default.
Use a fresh output path for another run; existing completed metrics are protected:

```bash
.venv/bin/python eval_stochastic_pusht.py --seed 43 \
  --output logs/eval_results/pusht_stochastic_seed43
```

For custom dynamics, use `eval.py --config-name=pusht_stochastic`, an explicit
checkpoint, `STABLEWM_HOME`, and overrides such as `world.force_std=4000` or
`world.angular_velocity_kick=10`. Save results in a fresh `output.dir`.

The suite writes resolved configs, checkpoint SHA-256, per-task diagnostics,
Wilson 95% intervals and paired rate differences. Each scenario also saves its
initial/goal images and `trajectories.npz`, containing states, block velocities,
actions, forces, and angular kicks. Array order is time, episode, component;
completed episodes freeze, and unexecuted action/force/kick slots can be NaN.
`executed_steps` determines the valid portion of each episode.

These are transfer tests for the existing checkpoint. They do not measure
whether a model trained on stochastic data predicts calibrated future modes.
