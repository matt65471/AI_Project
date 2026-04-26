# DQN / DRQN on Atari and MiniGrid

PyTorch implementations of **DQN** (Mnih et al. 2015) and **DRQN** (Hausknecht & Stone 2015), trained on Atari (Pong, Breakout) and MiniGrid Memory.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/matt65471/AI_Project.git
cd AI_Project
```

### 2. Create a virtual environment (Python 3.10+)

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Atari ROMs

Required for Pong and Breakout. Run once:

```bash
AutoROM --accept-license
```

If `AutoROM` isn't on your PATH:

```bash
python -m AutoROM --accept-license
```

### 5. (Optional) Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

CUDA is not required, but DRQN is much faster on GPU. The `train_light_drqn_minigrid.py` variant is tuned for CPU.

---

## Running

### Training

Each `train_*.py` is self-contained — hyperparameters are set as constants at the top. Run with:

```bash
python train_pong.py                  # DQN on ALE/Pong-v5
python train_atari.py                 # DQN on ALE/Breakout-v5
python train_minigrid.py              # DQN on MiniGrid-MemoryS7-v0
python train_drqn_minigrid.py         # DRQN on MiniGrid-MemoryS7-v0 (GPU)
python train_light_drqn_minigrid.py   # DRQN on MiniGrid-MemoryS7-v0 (CPU)
```

Checkpoints are written to `checkpoints/`, TensorBoard logs to `logs/`.

To monitor:

```bash
tensorboard --logdir logs
```

### Playing / recording videos

```bash
python play_pong.py
python play_breakout.py
python play_minigrid.py
```

Each script loads a checkpoint, runs a few episodes, and writes videos under `videos/`. Edit the `play(...)` call at the bottom of each script to point at a different checkpoint or change episode count.

---

## Default hyperparameters

All settings are constants at the top of each `train_*.py`. Edit there to change them.

### `train_pong.py` — DQN on ALE/Pong-v5

| Setting | Value |
|---|---|
| Total steps | 4,000,000 |
| Batch size | 32 |
| Replay capacity | 200,000 |
| Learning starts | 50,000 |
| Target update freq | every 10,000 updates |
| Discount γ | 0.99 |
| Learning rate (Adam) | 2.5 × 10⁻⁴ |
| ε start → end | 1.0 → 0.1 |
| ε anneal steps | 250,000 |
| Frame stack | 4 |
| Save interval | 100,000 |
| Seed | 32 |

### `train_atari.py` — DQN on ALE/Breakout-v5

| Setting | Value |
|---|---|
| Total steps | 500,000 |
| Batch size | 32 |
| Replay capacity | 20,000 |
| Learning starts | 10,000 |
| Target update freq | every 1,000 updates |
| Discount γ | 0.99 |
| Learning rate (Adam) | 1 × 10⁻⁴ |
| ε start → end | 1.0 → 0.1 |
| ε anneal steps | 500,000 |
| Frame stack | 4 |

### `train_minigrid.py` — DQN on MiniGrid-MemoryS7-v0

| Setting | Value |
|---|---|
| Total steps | 500,000 |
| Batch size | 32 |
| Replay capacity | 100,000 |
| Learning starts | 10,000 |
| Target update freq | every 1,000 updates |
| Discount γ | 0.99 |
| Learning rate (Adam) | 2.5 × 10⁻⁴ |
| ε start → end | 1.0 → 0.1 |
| ε anneal steps | 250,000 |
| Frame stack | 4 |
| Seed | 32 |

### `train_drqn_minigrid.py` — DRQN (GPU) on MiniGrid-MemoryS7-v0

| Setting | Value |
|---|---|
| Total steps | 500,000 |
| Batch size | 32 |
| Replay capacity | 500 episodes |
| Min episodes before training | 10 |
| Sequence length (BPTT) | 8 |
| LSTM hidden size | 512 |
| Target update freq | every 1,000 updates |
| Discount γ | 0.99 |
| Learning rate (Adam) | 2.5 × 10⁻⁴ |
| ε start → end | 1.0 → 0.1 |
| ε anneal steps | 250,000 |
| Frame stack | 4 |
| Seed | 42 |

### `train_light_drqn_minigrid.py` — DRQN (CPU) on MiniGrid-MemoryS7-v0

| Setting | Value |
|---|---|
| Total steps | 2,000,000 |
| Batch size | 32 |
| Replay capacity | 1,000 episodes |
| Min episodes before training | 100 |
| Sequence length (BPTT) | 16 |
| LSTM hidden size | 128 |
| Target update freq | every 10,000 updates |
| Update every N env steps | 8 |
| Discount γ | 0.99 |
| Learning rate (Adam) | 5 × 10⁻⁵ |
| ε start → end | 1.0 → 0.05 |
| ε anneal steps | 400,000 |
| Frame stack | 4 |
| Save interval | 5,000 |
| Seed | 51 |
---

## What each file does

| File | Purpose |
|---|---|
| `train_pong.py`, `train_atari.py` | DQN training on Atari Pong / Breakout |
| `train_minigrid.py` | DQN training on MiniGrid Memory (baseline) |
| `train_drqn_minigrid.py` | DRQN (LSTM) training on MiniGrid Memory |
| `train_light_drqn_minigrid.py` | DRQN tuned for CPU (smaller LSTM, larger update interval) |
| `play_pong.py`, `play_breakout.py`, `play_minigrid.py` | Load a checkpoint and record videos |
| `models/dqn_model.py` | Nature DQN convolutional Q-network |
| `models/drqn_model.py` | CNN → LSTM(512) → FC head |
| `models/light_drqn_model.py` | CPU-friendly CNN → LSTM(128) → FC head |
| `wrappers/atari_wrapper.py` | NoopReset, episodic life, frame skip, grayscale, resize, frame stack, reward clip |
| `wrappers/minigrid_wrapper.py` | RGB partial-observation rendering, resize, frame stack |
| `wrappers/light_minigrid_wrapper.py` | Same as above, lower-res for CPU |
| `buffers/episode_buffer.py` | Episodic replay buffer used by DRQN (stores full episodes, samples fixed-length subsequences) |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Saved model weights |
| `logs/` | TensorBoard event files |
| `videos/` | Output of `play_*.py` |