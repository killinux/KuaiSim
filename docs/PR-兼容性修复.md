# PR: Make KuaiSim run on Apple Silicon (MPS) and modern PyTorch

## Summary

This PR makes the KuaiRand benchmark pipeline run end-to-end on an Apple
Silicon Mac (M4 Pro) under PyTorch 2.8, with GPU acceleration via Metal
(MPS). It adds an MPS device path and fixes three pre-existing issues that
prevented the simulator/RL stages from running on a fresh, modern
environment. Verified end-to-end: data prep → user-response model training
→ whole-session environment → DDPG agent training.

## Motivation

On a CUDA-less machine (Apple Silicon) with current dependencies
(torch 2.8, numpy 2.x, pandas 2.x), the project could not run past
training the user model:

- training fell back to CPU only (no MPS path), ~2x slower than necessary;
- loading any trained checkpoint raised `UnpicklingError` (PyTorch 2.6
  `weights_only` default change);
- the RL entry (`train_actor_critic.py`) failed at import time twice
  (missing module + optional-env deps), and the environment's DataLoader
  hung on macOS.

## Changes

### 1. Apple Silicon MPS support  (commit `4eabca1`)
- `train_multibehavior.py`, `train_actor_critic.py`, `train_td3.py`,
  `train_general_model.py`, `train_online_policy.py`: device selection
  changed from `cuda → cpu` to `cuda → mps → cpu`. Purely additive `elif`;
  CUDA/CPU behavior unchanged.
- Recommended runtime flag: `PYTORCH_ENABLE_MPS_FALLBACK=1` (ops without an
  MPS kernel fall back to CPU instead of erroring).

### 2. `torch.load` compatibility with PyTorch ≥ 2.6  (commit `1737d08`)
- PyTorch 2.6 flipped `torch.load(weights_only=...)` to default `True`,
  which rejects these checkpoints because `reader_stats` embeds numpy
  scalars. This broke `get_user_model` (loading the user simulator into the
  RL environment) and every agent's load path.
- Pass `weights_only=False` on all `torch.load` calls for the project's own
  trusted checkpoints: `env/BaseRLEnvironment.py`,
  `env/KREnvironment_InfiniteRec_GPU.py`,
  `env/KRCrossSessionEnvironment_ModelBased.py`, `model/general.py`,
  and `model/agent/{A2C,TD3,BaseRLAgent,BaseOnlineAgent,OfflineSLAgent}.py`.

### 3. RL agent stage made importable & runnable  (commit `00c51b4`)
- **Broken import**: `model/agent/{BaseOnlineAgent,OnlineRerankAgent}.py`
  imported `model.agent.reward_func`, which does not exist; the reward
  functions live in `model/reward.py`. Repointed the imports.
- **Optional-env deps**: `from env import *` eagerly imported the
  VirtualTaobao / RecoGym / RecSim / RL4RS environments, which require
  external packages (GAN_SD, recogym, …). `env/__init__.py` now excludes
  those optional modules from `__all__`; the core KuaiRand envs import
  without those deps (import the specific module explicitly if you have
  them).
- **macOS DataLoader hang**: the GPU environments build
  `DataLoader(num_workers=8, pin_memory=True)` in `reset()`; on macOS this
  hangs (multiprocessing fork + pickling the large reader). Changed to
  `num_workers=0, pin_memory=False` in the KuaiRand envs (sampling one
  batch needs no workers; pin_memory is a no-op off-CUDA).

### 4. Docs  (commit `8f3071f`)
- `docs/KuaiSim-运行与原理总结.md`: end-to-end writeup (principle, remote
  deployment, data prep, these fixes, reproduction commands, results,
  demo, notebook usage).

## Verification

Run on Apple M4 Pro (14-core CPU / 20-core GPU, 64GB), macOS 26.4.1,
torch 2.8.0 (MPS), pandas 2.3.3, numpy 2.0.2:

- **Data prep**: 1,341,250 records / 19,574 users / 5,659 items.
- **User simulator** (`run_multibehavior.sh`, lr 1e-4, 10 epochs on MPS):
  loss 1.08 → 1.057; final AUC — is_click 0.726, long_view 0.729,
  is_like 0.840, is_follow 0.769.
- **MPS vs CPU**: ~470s vs ~1013s per epoch (~2.15x), with numerically
  equivalent validation AUCs.
- **Whole-session env**: loads the trained simulator, samples users, steps
  recommendations, temper-based leave + streaming replacement all work.
- **DDPG RL** (`train_actor_critic.py`, whole-session): trains actor/critic,
  accumulates reward, saves a loadable agent + `model.report`.

## Notes / Out of scope
- `model/recogym/agents/pytorch_mlr.py` still has hardcoded `.cuda()`
  (RecoGym path, unused by the KuaiRand benchmark); left untouched.
- `num_workers=0` is a single-machine-simulator-friendly default; on Linux
  with CUDA you may restore a higher value if desired.

## Commits
```
4eabca1  Add Apple Silicon MPS device support to training entry scripts
1737d08  Fix torch.load for PyTorch 2.6+ (weights_only default change)
00c51b4  Fix RL agent stage: broken import, optional-env deps, macOS DataLoader
8f3071f  Add docs: KuaiSim 运行与原理总结
```
