# Deployment

Takes a trained policy to the robot in four steps. Steps 1 to 3 are the same for every robot;
step 4 has a guide per robot.

| | step | script | what it tells you |
|---|---|---|---|
| 1 | export | `export_bundle.py` | turns the checkpoint into a **bundle** |
| 2 | validate | `validate_bundle.py` | the bundle computes the same observations and actions as training |
| 3 | rollout | `rollout_mujoco.py`, `rollout_viser.py` | the bundle tracks its motions in plain MuJoCo |
| 4 | sim2sim, then the robot | [g1/](g1/README.md), [x2/](x2/README.md) | the policy survives the robot's real control loop, then the robot |

A **bundle** is a folder with the policy's weights, its settings and the reference motions.
Running it needs only numpy and mujoco: no JAX, no LocoMuJoCo, nothing from training. That
is what lets the same files run on a robot's computer.

## 1. Export

```bash
python deployment/export_bundle.py --checkpoint outputs/<run>/PPOJax_saved.pkl --output deploy_bundles/<name>
```

For a dataset with many clips, add `--max-motions 20`.

## 2. Validate

```bash
python deployment/validate_bundle.py --bundle deploy_bundles/<name> --checkpoint outputs/<run>/PPOJax_saved.pkl
```

It must print PASS. Do this before trusting anything that follows: the bundle rebuilds the
observation by hand, and this is the check that it did so correctly.

Steps 1 and 2 need the training setup, since they rebuild the environment from the
checkpoint. Everything after does not.

## 3. Rollout

```bash
python deployment/rollout_mujoco.py --bundle deploy_bundles/<name> --viewer
```

```bash
python deployment/rollout_viser.py --bundle deploy_bundles/<name>
```

The first opens MuJoCo's viewer (`mjpython` instead of `python` on macOS), the second runs in
the browser with a clip picker and needs `loco-mujoco[viser]`. Policy and physics run in one
process on the model used in training, so a failure here is a bug in the export, not a
sim-to-real gap.

## 4. Sim2sim and the robot

> [!WARNING]
> Running a learned policy on a real robot is hazardous. A policy can fail without notice,
> and the robot can fall, hit people or objects, or damage itself. Everything in this
> repository is research code, provided as is and without warranty (see [LICENSE](../LICENSE)):
> you deploy at your own risk and are responsible for a safe setup. Hang the robot from a
> crane or gantry, keep people out of its reach, keep the emergency stop in hand, and run
> every policy in sim2sim first.

From here each robot has its own controller, its own simulated stand-in that speaks the
robot's middleware, and its own guide:

| robot | guide |
|---|---|
| Unitree G1 | [g1/README.md](g1/README.md) |
| AgiBot X2 Ultra | [x2/README.md](x2/README.md) |

In sim2sim the controller and the simulated robot are separate processes, so the controller
sees state that is already old, as on hardware. The same controller then drives the real
robot.

## More detail

[REFERENCE.md](REFERENCE.md) has the bundle format, the observation layout, the options of
the rollout, SuperMimic bundles, known quirks of the training stack and current limitations.
