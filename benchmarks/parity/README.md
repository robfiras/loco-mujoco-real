# Matched-task benchmarks against mjlab and MuJoCo Playground

Throughput and memory comparisons between frameworks are only meaningful if every framework runs the same task:
the same model, observations, rewards, commands, resets and terminations. This folder rebuilds two other
frameworks' G1 tasks from LocoMuJoCo components, on those frameworks' own models, so that a LocoMuJoCo step takes
and returns the same things as theirs, and only the implementation differs.

| Task | Rebuilt in | Original |
|---|---|---|
| MuJoCo Playground 0.2.0, `G1JoystickFlatTerrain` | `playground_g1.py` | `mujoco_playground/_src/locomotion/g1/joystick.py` |
| mjlab 1.6.1, `Mjlab-Velocity-Flat-Unitree-G1` (domain randomization events removed) | `mjlab_g1.py` | `mjlab/tasks/velocity/` |

Both run on all three LocoMuJoCo backends (CPU, MJX, MjWarp). The module docstrings list every component and the
few places where the implementation differs from the original without changing what a step computes.

## Files

- `playground_g1.py`: Playground's joystick task. Observations `state` (actor) and `privileged_state` (critic); the joystick command; 24 reward terms; pushes; Playground's randomized reset; its termination.
- `mjlab_g1.py`: mjlab's velocity task. The actor and critic groups with mjlab's noise; the twist command with heading control and its curriculum; 13 reward terms; per-substep contact-sensor bookkeeping; mjlab's reset and termination.
- `mjlab_model.py`: loads mjlab's exported model with its exact numbers.
- `mjlab_contacts.py`: mjlab's contact sensors on every backend. MJX computes them from its contact list.
- `export_models.py`: exports mjlab's compiled scene (full and feet-only contacts). Run in the mjlab environment.
- `record_playground.py`, `record_mjlab.py`: record the original tasks step by step. Run in the Playground or mjlab environment.
- `check_playground.py`, `check_mjlab.py`, `check_mjx_contacts.py`: equivalence checks against those recordings.
- `bench_parity.py`: throughput and memory benchmark of the matched tasks. Its protocol is that of the throughput table.

## Equivalence checks

The checks set the LocoMuJoCo state to the original framework's state before each recorded step and take the same
action. They then replace the physics by the original framework's own result of that step: the data after each
substep, and after mjlab's refresh. Everything the task computes from that data is then compared:

- the control targets;
- both observation groups;
- every reward term, and the reward;
- the termination;
- the task state after the step.

The physics itself cannot be compared over free rollouts. With 3 (Playground) or 10 (mjlab) solver iterations, the solver's early-stop test makes a step's result depend on rounding, so separately compiled programs diverge at some steps even in float64. Instead, the compiled models are compared field by field: they are identical except for names.

```bash
# Playground (record in an environment with playground==0.2.0, check in one with LocoMuJoCo)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python record_playground.py /tmp/pg 1 100 1.0
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python check_playground.py /tmp/pg cpu      # or mjx; mjwarp with a float32 recording
# mjlab (record in an environment with mjlab==1.6.1, on a GPU)
python export_models.py /tmp/mjlab_g1
python record_mjlab.py /tmp/ml 1 150 0.3 full
JAX_PLATFORMS=cpu python check_mjlab.py /tmp/ml full cpu                       # or mjx; mjwarp on a GPU
```

The checks and `bench_parity.py` read four environment variables:

- `LMJ_SRC` and `LMJ_REAL_SRC`: the source directories of LocoMuJoCo and of this extension.
- `PG_G1_ASSETS`: a directory holding Playground's G1 model files. These are the 65 files that `mujoco_playground._src.locomotion.g1.base.get_assets()` returns, written under their names.
- `MJLAB_G1_EXPORT`: the output directory of `export_models.py`.
