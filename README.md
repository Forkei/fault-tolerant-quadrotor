# Fault-tolerant quadrotor RL in simulation

A reimplementation of SimpleFlight (Chen et al., RAL 2025) on MuJoCo MJX with Crazyflie 2.1 parameters, extended with an RMA-style rotor-fault-tolerant policy and an early obstacle-avoidance policy.

**Status: prototype. Simulation only.** Nothing here has been run on a real vehicle, and no sim-to-real claim is made. Results are single-fault-type, few-seed numbers from my own evaluation scripts, not a benchmark.

The Python package is named `iron_man_drone` (the project's original codename).

---

## Milestones

| # | Goal | Status |
|---|---|---|
| M1 | SimpleFlight recipe on MJX, figure-eight tracking | Done. Passes the three gated trajectories. |
| M2 | Single-rotor fault tolerance via RMA two-phase training (privileged policy, then a causal history encoder) | Done against a loosened gate. The original spec targets were not met (see below). |
| M2.5 | Depth rendering (MJWarp) and obstacle infrastructure | Done. Infrastructure only, no new policy. |
| M3 | Joint fault-encoder and depth-based obstacle avoidance | Prototype. Avoids static pillars in most fair episodes; tracking precision and fault tolerance fall short of the targets. Not pursued further. |

Mapping and language-commanded flight (what I'd originally listed as M4/M5) were never started and are not part of this repo.

---

## M1 results

SimpleFlight on MuJoCo MJX, with the Crazyflie 2.1 parameters and the PPO hyperparameters from the paper's Table VI. Median Euclidean distance (MED) over 1000 steps, 3 seeds, GPU MJX evaluation (`eval_suite.py`), figure-eights started at the T/4 phase offset (see [Eval methodology](#eval-methodology)).

| Trajectory | Ours | Paper | Gate |
|---|---|---|---|
| figure_eight_slow (T=15 s) | 0.020 m | 0.016 m | <= 0.050 m, pass |
| figure_eight_normal (T=5.5 s) | 0.040 m | 0.028 m | <= 0.056 m, pass |
| figure_eight_fast (T=3.5 s) | 0.094 m | 0.051 m | <= 0.150 m, pass |
| pentagram_slow | 0.058 m | 0.024 m | not gated |
| polynomial (random) | 0.065 m | 0.032 m | not gated |
| zigzag (random) | 0.043 m | 0.052 m | not gated |

We are 1.3x to 1.8x above the paper on the figure-eights and 2.4x above on the pentagram. The gates are my own targets, derived from the paper's Table III, not pass criteria defined by the paper. An earlier CPU-MuJoCo evaluation gave numbers about 0.003 m lower (0.017 / 0.037 / 0.090 m) and polynomial/zigzag numbers that were too low because the first waypoint was pinned to the spawn point; both effects are explained in `notes/M1_3_results.md`.

---

## M2 results: RMA fault tolerance

Phase 1 trains an actor that is given the privileged perturbation vector e_t = [eta_1..eta_4, mass scale, F_x, F_y, F_z]. Phase 2 freezes that actor and trains a causal encoder phi(history) -> e_t from observable history, then deploys the actor with the encoder's estimate in place of the ground truth.

**Randomization during Phase 1:** one rotor at efficiency eta ~ U(0.5, 1.0) with probability 0.7 per episode (otherwise nominal), mass scale +/-20%, k_f +/-30%. Wind is not randomized (the F slots are always zero).

**Encoder:** flat MLP, 2300 -> 256 -> 128 -> 8 with tanh, input is the last 50 steps (0.5 s) of (observation, previous action). Trained offline on 20,480 rollouts of the Phase 1 policy; best validation MSE 0.0156 (normalized), 0.0115 averaged over the four rotor channels and 0.076 for mass scale, which is the weak channel.

Closed-loop evaluation, 3 seeds, MED in metres. "Fault" is rotor 0 at eta = 0.70. A `*` means at least one seed crashed.

| Trajectory | Phase 1 nominal | Phase 1 fault | Phase 2 nominal | Phase 2 fault |
|---|---|---|---|---|
| figure_eight_slow | 0.024 | 0.032 | 0.024 | 0.034 |
| figure_eight_normal | 0.057 | 0.079 | 0.057 | 0.081 |
| figure_eight_fast | 0.138 | 0.534* | 0.133 | 0.605* |
| pentagram_slow | 0.067 | 0.080 | 0.067 | 0.083 |
| pentagram_fast | 0.079 | 0.088 | 0.079 | 0.090 |
| polynomial | 0.088 | 0.733* | 0.087 | 0.141 |
| zigzag | 0.053 | 0.054 | 0.053 | 0.056 |

What this does and does not show:

- With the encoder in the loop, performance is within about 0.003 m of the privileged-state policy on every non-crashing condition. The history carries enough signal to identify a single degraded rotor.
- The fault penalty on figure_eight_normal is about 1.4x. On figure_eight_fast with a fault, 2 of 3 seeds crash for both phases; this is the edge of the fault envelope at this T/W, and it is not solved.
- The original M2 spec asked for nominal <= 0.037 m (M1 parity) and fault <= 0.060 m on figure_eight_normal. The Phase 1 policy gets 0.057 / 0.079 m. I relaxed the gate to 0.060 / 0.100 m after seeing this, so "passed" means passed against the relaxed gate. The nominal gap to M1 (0.040 m canonical) is mostly the cost of training with randomization.
- Closed-loop evaluation covers one fault only: rotor 0 at eta = 0.70. Other rotors, eta = 0.50 and combined mass-plus-rotor faults were not evaluated, and total rotor loss is not modelled.
- The encoder starts each episode with an all-zero history. Real-rollout analysis (`notes/M2_encoder_manifold_analysis.md`) suggests it then defaults to "no fault" for the first ~50 steps, which is the likely cause of the figure_eight_fast crashes. This was not fixed in M2.

Training: 15,000 PPO epochs, 2048 envs, roughly 3.75 h on a laptop RTX 4070. Phase 2 data collection took about 3 minutes and encoder training about 1 minute.

---

## M3 results: obstacle avoidance (prototype)

M2.5 adds an MJWarp depth camera (64x64) and a 16-slot obstacle randomizer on top of the unchanged MJX physics. M3 trains a policy from scratch with fault randomization and procedural pillar scenes at the same time, with the fault encoder trained jointly (no separate Phase 2).

```
Actor  (66-dim): [e^W (30), v (3), R (9), z_fault (8), depth bins (16)]
Critic (72-dim): [actor obs (66), k (1), 5 nearest obstacle distances (5)]
```

The depth image is min-pooled into 16 horizontal sectors. Scenes are `forest`, `urban`, `random` and `slalom` (training) with `hallway` held out. Reward is tracking plus a proximity penalty inside 0.5 m and a terminal crash penalty at 0.15 m. Run 1 used 1024 envs and 500M environment steps (epoch 15,258). Its frozen settings are in `experiments/m3_run1/config.yaml` (`w_track = 2.0`, `w_crash = 10.0`); note that the default `W_TRACK` in `quadrotor_env_m3.py` is now 6.0, so set `W_TRACK=2` to reproduce run 1.

Evaluation (`scripts/eval_m3.py`, 128 episodes per condition, greedy, 1000 steps). Episodes are bucketed by the clearance between the commanded path and the nearest obstacle. "Fair" means clearance >= 0.15 m. "Unfair" episodes command a path through the crash radius, so the policy can only survive by leaving the line.

| Metric | Result | Target |
|---|---|---|
| Collision-free, fair episodes, nominal | 92.3% | >= 80%, met |
| Tracking MED on collision-free episodes | 0.174 m | <= 0.10 m, **not met** |
| Collision-free, "unfair" episodes (deviate to survive) | 16% | no target |
| Collision-free, fair episodes, eta = 0.70 fault | 52.4% | >= 50%, marginal |
| Loss of attitude control (out of bounds), eta = 0.70 | 38.3% | weak |
| Figure-eight XY MED, clear air (run 1 final policy) | 0.287 m | M2 policy on the same task: 0.057 m |

The policy was already at these levels at 193M steps (38% of the schedule); the remaining 307M steps bought nothing, and tracking got slightly worse. A short diagnostic sweep (`notes/M3_v2_results.md`) found the terminal crash penalty, not the tracking weight, was making the policy conservative: lowering `W_CRASH` from 10 to 4 tightened figure-eight MED from 0.287 m to 0.215 m with fair collision-free rate unchanged (91.5%), while raising `W_TRACK` had no effect. Those sweep runs were short (about 15-20M steps) and I did not continue them. The hallway holdout was not evaluated. Fault handling in M3 is clearly worse than in M2.

---

## Architecture

### Action space

CTBR: collective thrust plus body rates (omega_x, omega_y, omega_z). The policy outputs a 4-dim Gaussian; a rate-PD controller with motor mixing and a first-order lag (tau = 25 ms) converts CTBR to per-rotor speeds.

### Observations

```
M1 actor   (42):  [e^W (30), v (3), R (9)]
M2 actor   (50):  [e^W (30), v (3), R (9), z (8)]
M2 critic  (51):  [e^W (30), v (3), R (9), e_t (8), k (1)]
```

- **e^W**: relative positions to the next 10 reference waypoints in the world frame, spaced 50 ms apart.
- **v**: world-frame linear velocity.
- **R**: body-to-world rotation matrix, flattened. Quaternions were tried early and did much worse on the figure-eight.
- **z**: 8-dim fault latent. In Phase 1 it is e_t from the simulator; at deployment it is the encoder output.
- **e_t**: privileged state [eta_1..eta_4, m_scale, F_x, F_y, F_z], seen by the critic.
- **k**: normalized episode step, critic only. Putting it in the actor caused failures on long flights.

Body rates and the previous action are left out of the actor.

### Networks and PPO

Actor and critic are separate Flax MLPs (3 x 256, ELU, LayerNorm) with separate Optax optimizers. PPO hyperparameters follow SimpleFlight Table VI unchanged: gamma 0.99, lambda 0.95, clip 0.2, actor LR 3e-4, critic LR 1e-4, 16 critic updates per actor update, entropy coefficient 1e-3, horizon 32, 8 minibatches, 5 epochs, 2048 envs (1024 for M3).

Rollout is a single `jax.lax.scan`, environments are batched with `jax.vmap` over a shared `mjx.Model`. Throughput is about 70k steps/s on a laptop RTX 4070 for M1/M2 and roughly 20-30k steps/s with depth rendering for M3.

### Trajectories

Trajectories are lazy (coefficients stored, positions evaluated on demand); precomputing per-env position arrays cost a 17x throughput loss under vmap.

Training mixes C2-continuous piecewise quintic polynomials with random zigzags. Evaluation trajectories (figure-eight at three speeds, pentagram at two) are never seen in training. An earlier polynomial generator produced zero-curvature segments and the policy could not turn the figure-eight apex; see L3 below.

### Domain randomization

Sampled at reset, constant within the episode. Rotor efficiency (one rotor, p = 0.7, eta ~ U(0.5, 1.0)), mass scale U(0.8, 1.2) applied as an extra force, thrust coefficient k_f +/-30%. Applied as `effective_speeds = rotor_speeds * sqrt(kf_mult * eta_i)`. At eta = 0.5 total thrust is 87.5% of nominal (T/W about 1.57).

---

## Eval methodology

SimpleFlight starts figure-eight references at t = T/4, where the lemniscate passes through (0, 0, 1), the drone's spawn point. This is not stated in the paper; it is in the authors' code (`omni_drones/envs/single/track.py`, `traj_t0 = 5.5 / 4`). Starting at t = 0 puts the reference 1 m away and adds about 0.03-0.04 m to the mean error. All numbers in this README use the T/4 offset and crash-only termination (height < 0.05 m, horizontal or vertical deviation > 5 m, tilt > 60 degrees).

Training-time inline evals in `train_m2.py` do not use the offset and should not be compared with the tables above. Evaluate with `scripts/eval_m1_suite.py`, `scripts/eval_m2_full.py` and `scripts/eval_m2_phase2.py`.

---

## Lessons from failed attempts

**L1: Resuming from a checkpoint without optimizer state wrecked a run.** Restoring params but reinitializing Adam moved MED from 0.085 m to 0.215 m immediately, followed by 11,000 epochs of crash-at-step-8 behaviour. Restore full optimizer state or don't resume.

**L2: Training reward hid a total failure on the held-out trajectory.** With 2048 random trajectories, mean reward can stay high while the figure-eight eval crashes every time. Use held-out MED as the convergence signal.

**L3: The polynomial generator was wrong for a month.** It produced straight segments with a smooth speed profile (curvature 0 everywhere), while the figure-eight apex needs about 4.8 1/m. The policy was fine on its curriculum and the curriculum was wrong. C2 piecewise quintics dropped MED from 0.105 m to 0.085 m in 2000 epochs.

Later entries (encoder startup, verifying numbers before reacting to them, a mis-specified regression gate in M3) are in `notes/lessons.md`.

---

## Setup (WSL2 / Linux, NVIDIA GPU)

```bash
bash scripts/setup_env.sh                 # one-time: JAX (CUDA 12) + MuJoCo MJX
pip install -e .                          # core dependencies
pip install -e ".[depth]"                 # only for M2.5/M3 (MJWarp depth rendering)
python -c "import jax; print(jax.devices())"

# M1
python scripts/train_m1.py
python scripts/eval_m1_suite.py

# M2 phase 1, then phase 2
python scripts/train_m2.py --config experiments/m2_phase1_baseline/config.yaml --num_envs 2048
python scripts/collect_phase2_data.py --checkpoint <phase1 run>/checkpoints/final
python scripts/train_phase2_encoder.py
python scripts/eval_m2_phase2.py

# M3 (needs the depth extra)
W_TRACK=2 python scripts/train_m3.py --run_name m3_run1
python scripts/eval_m3.py --ckpt ~/m3_checkpoints/m3_run1/final --n 128
```

Checkpoints and the 2.9 GB Phase 2 rollout dataset are not committed; the scripts above regenerate them. Training scripts refuse to start without a filled-in hypothesis note (`notes/M*_hypothesis.md`), a project rule. Stack: Python 3.12, JAX with CUDA 12, MuJoCo >= 3.1.6 (>= 3.8.1 with MJWarp), Flax, Optax, Distrax, Orbax.

---

## Repo layout

```
src/iron_man_drone/
  envs/
    crazyflie.xml, crazyflie_depth.xml   MJCF models (forces applied programmatically)
    quadrotor_env.py          M1/M2 MJX env: reset/step/obs/reward, DR, privileged state
    quadrotor_env_depth.py    M2.5: adds obstacle slots and MJWarp depth rendering
    quadrotor_env_m3.py       M3: depth bins, history buffer, obstacle reward, scene sampling
    trajectories.py           lazy trajectories: quintic poly, zigzag, figure-eight, pentagram
  control/ctbr_controller.py  CTBR -> rotor speeds (rate PD, X mixer, motor lag)
  policy/
    networks.py               Actor and Critic
    encoder.py                causal history encoder, e_t normalization
    ppo.py                    PPO with lax.scan rollout, separate actor/critic states
  evaluation/eval_suite.py    shared eval harness (T/4 offset, crash-only termination)
  utils/
    domain_randomization.py   rotor/mass/k_f sampling
    obstacle_randomization.py, scene_generator.py   pillar scenes for M2.5/M3
experiments/                  configs, training logs, small result files (no checkpoints)
notes/                        hypothesis docs, specs, results, lessons
scripts/                      train_*, eval_*, diag_*, validate_*, smoke tests, benchmarks
```

---

## Dynamics (Crazyflie 2.1)

| Parameter | Value |
|---|---|
| Mass | 0.0321 kg |
| Inertia (Ixx, Iyy) | 1.4e-5 kg m^2 |
| Inertia (Izz) | 2.17e-5 kg m^2 |
| Thrust coefficient k_f | 2.35e-8 N/(rad/s)^2 |
| Torque coefficient k_m | 7.24e-10 N m/(rad/s)^2 |
| Motor time constant | 25 ms |
| Arm length | 46 mm (35.4 mm to rotor center, X-config) |
| Max rotor speed | 2315 rad/s |
| Sim frequency | 100 Hz |
| Episode length | 10 s (1000 steps) |

These follow the Crazyflie 2.1 parameters used by SimpleFlight. The M1/M2 simulation has no aerodynamic effects, sensor noise, latency or battery model.

---

## Papers

- **SimpleFlight**: Chen et al., RAL 2025. arXiv:2412.11764. Source of the M1 architecture, hyperparameters and evaluation setup.
- **RMA**: Kumar et al., RSS 2021. arXiv:2107.04034. Two-phase privileged training used in M2.
- **MAVEN**: arXiv:2603.10714. Source of the rotor-fault and mass randomization ranges.
- **PureJaxRL**: Lu et al., 2022. The fully JIT-compiled PPO pattern used throughout.

---

## Contact

Olivier Couthaud, olivier.couthaud@gmail.com, [@olivier_couth](https://x.com/olivier_couth)

If you work on agile drone RL or fault-tolerant control and want to compare notes, I'd like to hear from you.
