# maniskill_stackcube

## StackCube Migration Notes

This repo was migrated from PushCube-v1 to StackCube-v1. Key dimension and
target changes (versus the PushCube version):

```text
env_id:           PushCube-v1   -> StackCube-v1
robot_uids:       panda         -> panda_wristcam (StackCube default)
target actor:     base_env.obj  -> base_env.cubeA (red cube being stacked)
obs dim (state):  35            -> 48
force feature:    1 angle (atan2(fy,fx))         -> 3 discrete (fx, fy, fz)
step input dim:   36 (35 obs+1 angle)            -> 51 (48 obs+3 force)
seq input dim:    324 (9*36)                     -> 459 (9*51)
action grid:      121 (11 angles on unit circle) -> 1331 (11^3 = fx,fy,fz ∈ {-1,...,1} step 0.2)
```

Force sampling: each of fx, fy, fz is independently chosen from
{-1.0, -0.8, ..., 0.8, 1.0} (11 values per axis). The unit vector is
NOT normalized; it is multiplied directly by `force_mag` before being
passed to `cubeA.apply_force`. The criticality model receives the
pre-scale unit force (3 values) as its per-step force feature.

Affected files (all hardcoded shape values updated):

```text
criticality/utils/criticality_model.py   (Reward_Model input_dim 36 -> 51)
criticality/utils/dqn.py                 (input assert 324 -> 459, view 9,36 -> 9,51,
                                          self.angles (121) -> self.force_grid (1331,3))
criticality/stage2/stage2_process.py     (STEP_DIM 51, 324 -> 459, no angle conversion)
criticality/stage2/stage2_train.py       (Reward_Model(input_dim=51))
criticality/stage2/stage2_collect.py     (records fx, fy, fz per step)
criticality/test/maniskill_ordinary_nade.py
    (NADEConfig.base_feature_dim 35 -> 48,
     calcu_q tensor shapes -> (8,51)/(1331,9,51),
     idx_to_action: angle on unit circle -> direct (fx,fy,fz) grid lookup,
     step input_buffer: state + [angle] -> state + [fx, fy, fz],
     env.unwrapped.obj.apply_force -> env.unwrapped.cubeA.apply_force)
examples/baselines/ppo/ppo.py             (training + eval disturbance use 3D discrete grid)
criticality/stage1/stage1_collect.py     (env_id default, cube_actor)
criticality/stage1/stage1_train.py       (dataset / save paths)
criticality/stage2/stage2_collect.py     (env_id default, import path)
criticality/test/test_model.py           (env_id default)
```

All absolute data / model paths were replaced with StackCube placeholders
under `/mnt/mnt1/tyy/ManiSkill_stackcube/` and
`/home/teamcommon/tyy/ManiSkill_stackcube/`, each marked with a `TODO`
comment so they can be pointed at the actual machine paths before running.

The Reward_Model.forward() shape logic was changed to drive its
chunk-size from `self.input_dim`, so future input_dim changes only
require updating the constructor default.

External forces are applied only to `cubeA` (red cube). Force probability
and grid-size settings are unchanged (force_prob=0.3, grid_size=11).

## Current Experiment Files

```text
criticality/stage1/
  nde.py
  stage1_collect.py
  stage1_train.py

criticality/stage2/
  stage2_collect.py
  stage2_process.py
  stage2_train.py

criticality/test/
  maniskill_ordinary_nade.py
  test_model.py

training/
  ppo_offline.py

utils/
  criticality_model.py
  data_utils.py
  dqn.py
  reward_model.py
```


The original PPO baseline file is still kept in the ManiSkill project
structure:

```text
examples/baselines/ppo/ppo.py
```

## 0. Original Policy

The original PPO policy is trained or loaded through:

```text
examples/baselines/ppo/ppo.py
```

Later NDE, NADE, and offline PPO data-collection scripts load a policy through
the `--checkpoint` argument. The checkpoint can be either an original PPO
checkpoint or an offline PPO checkpoint saved in `training/models/`, for example:

```text
training/models/offline_model_ep60.pt
```

In short:

```text
Original policy code: examples/baselines/ppo/ppo.py
Policy loading argument: --checkpoint
Available offline policy checkpoints: training/models/offline_model_ep*.pt
```

## 1. Stage 1: NDE Data Collection and First-Stage Model Training

Stage 1 has two main purposes:

```text
1. Collect NDE data with random external forces.
2. Train the first-stage criticality / reward model from the collected data.
```

Files used:

```text
criticality/stage1/stage1_collect.py
criticality/stage1/stage1_train.py
utils/criticality_model.py
utils/data_utils.py
utils/reward_model.py
```

### criticality/stage1/stage1_collect.py

This script loads a policy, runs the StackCube environment, applies random
external forces to the cube, and collects trajectories.

Main responsibilities:

```text
Load a policy checkpoint(initial model trained in ppo.py).
Run StackCube-v1.
Apply random external forces.
Record whether an episode reached success_once.
Separate successful and failed trajectories.
```

Common arguments:

```text
--checkpoint      Policy checkpoint to load.
--env_id          Environment name, such as StackCube-v1.
--obs_mode        Observation mode, such as state.
--control_mode    Control mode, such as pd_joint_delta_pos.
--force_mag       External force magnitude.
--force_prob      Probability of applying the external force.
--n               Number of episodes to collect.
```

### criticality/stage1/nde.py

This is another NDE data-collection script. It is almost identical with stage1_collect.py.
Its purpose is to test new-trained model and save data.(which is used after ppo_offline.py)
It saves each episode with the
following fields:

```text
obs
action
reward
force
success
```

Here, `success` is the episode-level success label.

### criticality/stage1/stage1_train.py

This script trains the first-stage criticality / reward model.

It depends on the model and data utilities in:

```text
utils/criticality_model.py
utils/data_utils.py
utils/reward_model.py
```

If the training script contains dataset paths, check and update them according
to the data location on the current machine before running it.

## 2. Stage 2: NADE Collection and Criticality Update

Stage 2 replaces uniform random force sampling with NADE sampling. A
criticality model is used to estimate which force directions are more dangerous,
and the force direction is sampled from the resulting NADE distribution.

Files used:

```text
criticality/stage2/stage2_collect.py
criticality/stage2/stage2_process.py
criticality/stage2/stage2_train.py
criticality/test/maniskill_ordinary_nade.py
criticality/test/test_model.py
utils/criticality_model.py
utils/dqn.py
```

### criticality/test/maniskill_ordinary_nade.py

This is the core NADE environment wrapper (moved to `criticality/test`).

Main responsibilities:

```text
Load the criticality model.
Evaluate candidate force directions.
Build the NADE sampling distribution from criticality scores.
Apply the sampled force to the cube.
Record the corresponding importance-sampling weight.
```

### criticality/stage2/stage2_collect.py

This script collects data in the NADE environment, using the criticality model trained in stage1.

It calls the NADE environment wrapper now located at `criticality/test/maniskill_ordinary_nade.py`.

It also loads a policy checkpoint and a criticality model checkpoint.

### criticality/stage2/stage2_process.py

This script processes Stage 2 collection batches and converts the raw batch
data into the replay-buffer format used by later training.

### criticality/stage2/stage2_train.py

This script trains or updates the Stage 2 criticality model.

It uses:

```text
utils/criticality_model.py
utils/dqn.py
```

## 3. Offline PPO / Continual Learning

The continual-learning pipeline has two steps:

```text
1. Collect offline PPO data with criticality/test/test_model.py.
2. Train an offline PPO policy with training/ppo_offline.py.
```

Files used:

```text
criticality/test/test_model.py
criticality/test/maniskill_ordinary_nade.py
training/ppo_offline.py
training/models/offline_model_ep*.pt
```

### criticality/test/test_model.py

This script collects offline PPO training data (moved to `criticality/test`).

It uses:

```text
Policy checkpoint
Criticality model checkpoint
NADE environment wrapper from criticality/test/maniskill_ordinary_nade.py
```

The saved data usually contains:

```text
obs
actions
weights
rewards
dones
log_probs
```

### training/ppo_offline.py

This script trains offline PPO from the data collected by `criticality/test/test_model.py`.

It supports one or more dataset directories through `--dataset`, and it loads
the initial policy through `--initial_ckpt`.

After `ppo_offline.py` finishes training, the newly saved policy checkpoint can
be used again in `criticality/test/test_model.py` as the next `--checkpoint` for iterative
data collection and offline PPO training.

The available offline PPO checkpoints in this repository are:

```text
training/models/offline_model_ep*.pt
```

## 4. NDE and NADE Testing Files

### NDE Testing

NDE testing uses random external force disturbance without the NADE criticality
distribution.

Files used:

```text
criticality/stage1/stage1_collect.py
criticality/stage1/nde.py
```

Conceptually:

```text
NDE = randomly sampled force direction
```

### NADE Testing

NADE testing uses the criticality model to bias the force sampling
distribution.

Files used:

```text
criticality/test/maniskill_ordinary_nade.py
criticality/stage2/stage2_collect.py
criticality/test/test_model.py
```

Conceptually:

```text
NADE = force direction sampled according to criticality model output
```

## 5. File Relationship Summary

```text
Original PPO policy:
  examples/baselines/ppo/ppo.py

Stage 1 NDE data collection:
  criticality/stage1/stage1_collect.py
  criticality/stage1/nde.py

Stage 1 criticality / reward model training:
  criticality/stage1/stage1_train.py
  utils/criticality_model.py
  utils/data_utils.py
  utils/reward_model.py

Stage 2 NADE environment:
  criticality/test/maniskill_ordinary_nade.py

Stage 2 NADE data collection:
  criticality/stage2/stage2_collect.py

Stage 2 data processing:
  criticality/stage2/stage2_process.py

Stage 2 criticality model update:
  criticality/stage2/stage2_train.py
  utils/criticality_model.py
  utils/dqn.py

Offline PPO data collection:
  criticality/test/test_model.py
  criticality/test/maniskill_ordinary_nade.py

Offline PPO training:
  training/ppo_offline.py
  training/models/offline_model_ep*.pt
```

## 6. Running Notes

Run commands from the repository root:

```bash
cd ManiSkill_new
```

When comparing different experiments, keep the following settings consistent:

```text
env_id
obs_mode
control_mode
force_mag
force_prob
checkpoint
criticality_ckpt
```

For long-running jobs, use:

```bash
nohup python -u path/to/script.py ... > run.log 2>&1 &
```
