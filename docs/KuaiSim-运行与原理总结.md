# KuaiSim 运行与原理总结

> 本文档记录在 Apple Silicon Mac（M4 Pro）上**从零把 KuaiSim 完整跑通**的全过程：原理、环境、数据准备、为新版 PyTorch/Apple GPU 做的兼容性修复、完整复现命令、训练结果，以及"模拟用户"的实际演示。
>
> 生成日期：2026-06-13
>
> **状态：✅ 全链路在远程 M4 Pro 上跑通** —— 数据预处理 → 用户模拟器训练（lr0.0001 满 10 epoch）→ 环境演示 → RL 阶段（DDPG 整会话）smoke，全部通过；3 个兼容性修复已合入 GitHub `main`。

---

## 1. 这个项目是什么

**KuaiSim** 是快手开源的、面向**强化学习推荐系统（RL-based Recommendation）的在线模拟器与基准测试平台**（论文 arXiv:2309.12645）。

核心思想：推荐系统的 RL 算法难以直接在真实线上做实验（成本高、风险大），于是用真实日志（KuaiRand 数据集）训练一个**会模仿真人反应的"虚拟用户"**，让推荐算法在这个虚拟用户上反复试错、训练——相当于推荐领域的 Gym 环境。

### 1.1 两步训练范式

| 步骤 | 训练对象 | 作用 |
|------|----------|------|
| **第一步** | 用户响应模型 `KRMBUserResponse` | 学会预测用户对视频的 7 种行为概率，之后被**冻结当作"环境/虚拟用户"** |
| **第二步** | RL 推荐策略（DDPG/TD3/A2C/HAC…） | 把用户模型当环境，训练"推什么列表能让用户满意且久留" |

### 1.2 用户响应模型结构（KRMBUserResponse）

```
用户侧:  user_id 嵌入 + 13 个画像特征(活跃度/粉丝段/注册天数…) → 用户向量
物品侧:  video_id 嵌入 + 4 个视频特征(类型/音乐/上传方式/标签) → 物品向量
历史序列: [历史视频编码 + 当时反馈编码] → Transformer(因果掩码) → 用户当前状态
打分:    用户状态 → DNN 生成 7 个行为的注意力核，与候选物品点积 → 7 个行为分数
损失:    7 种行为分别 BCE，按行为权重加权求和；稀有正反馈用负采样率重加权
```

预测的 7 种行为：`is_click, long_view, is_like, is_comment, is_forward, is_follow, is_hate`

### 1.3 模拟器作为 RL 环境

| RL 概念 | 对应 |
|---------|------|
| State | 用户画像 + 历史编码（用户模型给出） |
| Action | 推荐列表 slate（默认 6 个视频） |
| Reward | 用户模型预测的 7 种行为按权重加权求和 |
| Done | 用户"耐心值 temper"耗尽则离开（推得越差扣得越多）|

### 1.4 三个 benchmark 层次

```
单次请求(列表推荐)  →  整个会话(整会话推荐)  →  跨会话(留存优化)
CF/ListCVAE/PRM        DDPG/TD3/A2C/HAC          TD3/RLUR/CEM
```

### 1.5 用户分布怎么做的

- **谁出现、多频繁**：环境 `reset()` 用 `DataLoader(reader, shuffle=True)` 对**请求行**均匀采样 = 对用户**按真实请求条数加权** → 还原线上"重度用户主导"的长尾流量，而非对 19574 个用户等概率。
- **用户是谁**：每条样本带真实 user_id + 画像 + **因果截断历史**（只看该时刻之前、最多 100 条）。
- **何时走**：会话内 temper 模型，离开时机由推荐质量内生决定（对应 Depth 指标）。
- **何时回**：跨会话 return-day 用几何分布 / 留存模型预测。
- **流式补充**：32 个用户并行，谁离开立刻补新用户，维持稳态流量。

---

## 2. 远程运行环境

| 项 | 值 |
|----|----|
| 机器 | Mac mini, **Apple M4 Pro**（14 核 CPU = 10P+4E，20 核 GPU），64GB 统一内存 |
| 系统 | macOS 26.4.1, arm64 |
| 部署目录 | `/Users/bytedance/work/hn/kuai/` |
| Python | venv（系统 python3.9.6）|
| 关键依赖 | **torch 2.8.0（MPS=True, CUDA=False）**, pandas 2.3.3, numpy 2.0.2 |

> ⚠️ Apple Silicon 没有 NVIDIA CUDA。原始代码只判断 `torch.cuda.is_available()`，没有 Apple **MPS**(Metal) 路径，默认只能跑 CPU。本次做了 MPS 适配（见 §4）。

### 2.1 环境搭建

```bash
cd /Users/bytedance/work/hn/kuai
python3 -m venv venv
source venv/bin/activate
pip install torch pandas numpy scikit-learn tqdm matplotlib
# notebook 演示额外需要：
pip install jupyter notebook ipykernel
```

---

## 3. 数据预处理

**坑**：仓库自带的原始数据 ≠ 训练脚本要读的数据，必须先预处理。

- 原始数据：`dataset/kuairand/kuairand-Pure/data/`
  - `log_standard_4_08_to_4_21_pure.csv`、`log_standard_4_22_to_5_08_pure.csv`、`user_features_pure.csv`、`video_features_basic_pure.csv`
- 训练脚本要读：`code/dataset/Kuairand_Pure/`
  - `log_session_4_08_to_5_08_Pure.csv`、`user_features_Pure_fillna.csv`、`video_features_basic_Pure_fillna.csv`

官方做法是跑 `code/preprocess/KuaiRandDataset.ipynb`。本次把其中**真正产出文件的几步**抽成了独立脚本 `preprocess.py`（远程 `/Users/bytedance/work/hn/kuai/preprocess.py`）：

1. 合并两段日志
2. `run_multicore(n_core=20)` 做 20-core 过滤
3. 构建 session/position、按 (user, time) 排序、重算 date → 存 `log_session`
4. user / video 特征 fillna

**预处理结果**：过滤后 **1,341,250 条记录、19,574 用户、5,659 视频**；切分为 train/val/test = `1,124,729 / 89,160 / 89,160`（按每用户留最后 5+5 条做 val/test 的时序划分）。

---

## 4. 为本机做的兼容性修复（重要）

代码是 2023 年写的，在 torch 2.8 + Apple Silicon 上有两处需要改。**已提交并推送到 `main`。**

### 4.1 MPS 设备支持（commit `4eabca1`）

5 个训练入口脚本（`train_multibehavior.py / train_actor_critic.py / train_td3.py / train_general_model.py / train_online_policy.py`）的设备选择，从 `cuda→cpu` 改为 `cuda→mps→cpu`：

```python
if args.cuda >= 0 and torch.cuda.is_available():
    device = f"cuda:{args.cuda}"
elif args.cuda >= 0 and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"                      # 新增：Apple GPU
else:
    device = "cpu"
```

运行时建议设 `export PYTORCH_ENABLE_MPS_FALLBACK=1`，个别没有 MPS 内核的算子自动回退 CPU，不崩溃。

**实测加速**：单 epoch CPU 1013s → **MPS 470s，约 2.15x**；且训练后验证 AUC 与 CPU **数值等价**（加速不损失正确性）。

### 4.2 torch.load 兼容 PyTorch 2.6+（commit `1737d08`）

PyTorch 2.6 把 `torch.load` 的 `weights_only` 默认从 `False` 改成 `True`，而本项目 checkpoint 里 `reader_stats` 含 numpy 标量，会被拒绝加载——**这会让第二步 RL 训练加载用户模型直接失败**（`get_user_model`）。

修复：给核心路径里加载**本项目自有可信 checkpoint** 的所有 `torch.load` 加 `weights_only=False`，涉及 9 个文件（`env/BaseRLEnvironment.py`、`model/general.py`、`model/agent/*` 等）。

---

## 5. 完整跑通步骤（可复现）

```bash
cd /Users/bytedance/work/hn/kuai/code
source ../venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ① 数据预处理（一次性）
python ../preprocess.py

# ② 训练用户模拟器（第一步）。脚本内 --cuda 0 现在会自动走 MPS
bash run_multibehavior.sh
#   产出: output/Kuairand_Pure/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.checkpoint
#         + 同名 .model.log（RL 阶段通过 --uirm_log_path 引用）

# ③ 训练 RL 推荐策略（第二步），三选一：
bash train_ddpg_krpure_wholesession.sh     # 整会话
bash train_listcvae_krpure_request.sh       # 列表推荐
bash train_td3_kpure_crosssession.sh        # 留存优化
```

---

## 6. 训练结果

### 6.1 用户模拟器（lr=0.0001，完整 10 epoch）AUC 随训练上升

| 行为 | Epoch 1 | Epoch 6 | **Epoch 10（最终）** |
|------|---------|---------|---------------------|
| is_click | 0.688 | 0.717 | **0.726** |
| long_view | 0.689 | 0.720 | **0.729** |
| is_like | 0.722 | 0.808 | **0.840** |
| is_comment | 0.664 | 0.698 | **0.707** |
| is_forward | — | 0.642 | **0.655** |
| is_follow | 0.679 | 0.733 | **0.769** |
| is_hate | 0.611 | 0.658 | **0.634** |

约 700s/epoch on MPS；loss 从 ~1.08 稳步降到 **1.057**。最终模型：
`output/Kuairand_Pure/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.checkpoint`（RL 阶段即用它）。

> 官方脚本还会接着训第二个学习率（0.00001）作为额外超参变体，RL 阶段用不到，本次为节省机器在 lr0.0001 完成后停止。

### 6.2 RL 阶段（DDPG 整会话）smoke 验证

用训好的用户模型当环境，跑 DDPG（`train_actor_critic.py`，整会话）冒烟，验证第二步端到端可用：

```
Setup critic: QCritic(...)  /  buffer: HyperActorBuffer  /  agent: DDPG
Total 50 prepare steps → Training:
step 10 @ online episode: step(depth)=14.7, leave=1.4, coverage=74.8, ILD=0.985
        @ training: actor_loss=-0.747, critic_loss=0.256, Q=0.703,
                    avg_total_reward=9.92 (max 10.83)
                    is_click=0.67 long_view=0.65 is_like=0.61 …
RL_SMOKE_EXIT=0
```

产出可加载的 agent：`output/Kuairand_Pure/agents/SMOKE_DDPG/` 下 `model_actor`(2.1M)、`model_critic`、各 optimizer、`model.report`（`TrainingCurves.ipynb` 可读此文件画曲线）。

> 即：用户模拟器被成功当作 RL 环境，DDPG 的 actor/critic 完成真实梯度更新、reward 正常累积、模型正常落盘——**第二步 RL 流水线打通**。完整 benchmark 只需把 `--n_iter` 调回 20000 即可。

---

## 7. "模拟用户"实际演示

用训练好的用户模型实例化 `KREnvironment_WholeSession_GPU`，`reset()` 抽一批真实用户，连续喂随机推荐看反馈与离开：

**抽 8 个真实用户**，每个带画像（活跃度/粉丝段/注册天数/onehot 特征，全部 one-hot）和历史序列（`history (8,100)` + 各特征 + 7 种历史反馈）。连推 6 步随机推荐：

```
候选视频池 5659，每次推 6 个
step1: temper=8.62 | 离开 0 | coverage=47 ILD=0.986
step2: temper=7.21 | 离开 0
step3: temper=5.90 | 离开 0
step4: temper=4.58 | 离开 0
step5: temper=3.23 | 离开 0
step6: temper=4.21 | 离开 2   ← 2 个用户耐心耗尽离开，被新用户补上，均值回升
       (每步还给出 7 种行为平均概率、coverage、ILD)

反馈字典 keys: ['immediate_response', 'user_state', 'coverage', 'ILD', 'done']
  immediate_response: (8, 6, 7)   = (用户数, 推荐位, 7种行为)
  user_state:         (8, 1, 192) = RL 的状态向量
```

可见：**耐心值随交互递减（推得越差扣得越多），耗尽即离开，离开者被新用户替换**——这正是 §1.5 描述的"会话内离开 + 流式补充"机制在真实运行。

> ⚠️ **macOS 关键修复**：环境 `reset()` 里 `DataLoader(num_workers=8)` 在 macOS 上会卡死（多进程 fork + 序列化大 reader）。演示脚本里 monkeypatch 成 `num_workers=0` 后秒级跑通。RL 阶段同理需要此修复（已在 env 文件中改为 0）。

---

## 8. Notebook 怎么用

> 前提：在 `code/` 目录、用上面的 venv 起 Jupyter（推荐在远程起服务、本地浏览器经 SSH 端口转发访问，因为数据/模型都在远程）。

| Notebook | 用途 | 关键改动 |
|----------|------|----------|
| `preprocess/KuaiRandDataset.ipynb` | 数据预处理 + EDA | 改 `data_path`；产出文件的是最后几个 cell |
| `WholeSessionRecEnvironment.ipynb` | 整会话环境交互调试（不训练）| 改 `uirm_log_path` 指向你训练好的用户模型 `.model.log` |
| `SlateRecEnvironment.ipynb` | 列表环境调试 | 同上 |
| `CrossSessionEnvironment.ipynb` | 跨会话/留存环境调试 | 同上 |
| `TrainingCurves.ipynb` | 画 RL 训练曲线 | 改 `expe` 为 agent 实验文件夹名，读 `model.report` |

**Plan A：远程跑 Jupyter，本地浏览器访问（推荐）**

```bash
# 远程启动（在 code/ 目录、venv 内）
cd /Users/bytedance/work/hn/kuai/code && source ../venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
jupyter notebook --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.token=kuaisim

# 本地建 SSH 端口转发
ssh -N -L 8888:127.0.0.1:8888 bytedance@<远程IP>
# 然后本地浏览器打开： http://localhost:8888/tree?token=kuaisim
```

`WholeSessionRecEnvironment.ipynb` 的 `uirm_log_path` 已指向训好的模型，从上往下跑即可看到模拟用户与反馈。

---

## 9. 注意事项 / 已知特性

- **只用得上 CPU+MPS**：那块 20 核 GPU 通过 MPS 用上了，但加速比受限于小 batch（128）和小模型维度，约 2x（非理论 3-5x）。
- **macOS DataLoader 慢启动**：环境 `reset()` 用 `num_workers=8` 的 DataLoader，在 macOS 上 fork 多进程 + 序列化大 reader 较慢，多任务争抢时尤其明显。
- **RecoGym 子环境跑不了**：`model/recogym/agents/pytorch_mlr.py` 有 4 处硬编码 `.cuda()`，本机会崩；但不影响主线 KuaiRand 三个 benchmark。
- **完整官方脚本**跑 2 个学习率 × 10 epoch ≈ 2.7h；RL 阶段只需 lr0.0001 那个模型。

---

## 10. 关键路径速查

```
远程: /Users/bytedance/work/hn/kuai/
  ├── code/                         源码（= GitHub main，含本次修复）
  │   ├── dataset/Kuairand_Pure/    预处理产出
  │   ├── output/Kuairand_Pure/env/ 用户模型 checkpoint + log
  │   └── run_multibehavior.sh 等   训练脚本
  ├── preprocess.py                 数据预处理脚本（替代 notebook）
  └── venv/                         Python 环境

GitHub: killinux/KuaiSim @ main
  4eabca1  MPS 设备支持
  1737d08  torch.load weights_only 修复
```
