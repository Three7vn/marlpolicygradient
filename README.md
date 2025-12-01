# Multi-Agent Deep Deterministic Policy Gradient for Tennis

An implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for the Unity ML-Agents Tennis environment, featuring advanced reinforcement learning techniques.

## Project Overview

This project trains two cooperative agents to play tennis, keeping the ball in play for as long as possible.

<p align="center">
  <img src="animation.gif" alt="Trained Tennis Agents" width="500"/>
</p> 

## Key Features

- **Multi-Agent Cooperative Learning**: Two agents learn to collaborate
- **Distributional RL**: C51-style value distribution learning for better value estimation
- **Prioritized Experience Replay**: Focuses learning on important transitions
- **Noisy Networks**: Parameter space noise for structured exploration
- **N-step Returns**: Multi-step bootstrapping for faster credit assignment
- **Soft Target Updates**: Polyak averaging for stable learning

## 🏗️ Project Structure

```text
selfdrivingmarl/
├── agents/
│   ├── agent.py              # DDPG agent implementation
│   ├── agent_utils.py        # Checkpointing utilities
│   └── __init__.py
├── learning/
│   ├── hp.py                 # Hyperparameters configuration
│   ├── model.py              # Actor-Critic networks
│   ├── multistep_buffer.py   # Prioritized replay buffer
│   └── __init__.py
├── Unity/
│   └── Tennis.app            # Unity Tennis environment (optional path)
├── results/                   # Training outputs (checkpoints, scores)
├── tennismaddpg.ipynb        # Main training notebook
├── training.py               # Training loop implementation
├── wrapper.py                # Training wrapper for notebooks
├── ddpg_wrapper.py           # Agent wrapper for notebooks
└── requirements.txt
```

## 📋 Requirements

- Python 
- PyTorch
- NumPy
- Matplotlib
- Unity ML-Agents (mlagents-envs or unityagents)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: Download the [Unity Tennis environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip), extract it, and place it in a `Unity/` folder at the project root.

## Environment

The Tennis environment consists of two agents controlling rackets to bounce a ball over a net.

**Observation Space**: 24 dimensions per agent (position and velocity of racket and ball)

**Action Space**: 2 continuous actions per agent

- Movement toward/away from net
- Jumping

**Rewards**:

- +0.1 for hitting ball over net
- -0.01 for letting ball hit ground or go out of bounds

**Success Criterion**: Average score ≥ 0.5 over consecutive episodes

## Usage

### Training

Open and run `tennismaddpg.ipynb` in Jupyter:

```bash
jupyter notebook tennismaddpg.ipynb
```

The notebook provides:

1. Environment exploration
2. Hyperparameter configuration
3. Agent training
4. Performance visualization
5. Model evaluation

### Results

Training artifacts are automatically saved to `results/`:

- `checkpoint_solved_*.pth`: Model when environment is solved
- `checkpoint_final_*.pth`: Final model after training
- `training_scores_*.npy`: Episode scores history

## Algorithm Details

The implementation uses DDPG with several enhancements:

1. **Actor-Critic Architecture**: Separate networks for policy (actor) and value (critic)
2. **Distributional Learning**: Models full distribution of Q-values using C51
3. **Noisy Networks**: Learnable exploration through parameter-space noise
4. **Prioritized Replay**: Samples important transitions more frequently
5. **N-step Returns**: Looks ahead multiple steps for better credit assignment

## Performance

The agent typically solves the environment (average score ≥ 0.5) within 150-200 episodes.

## Hyperparameters

Key hyperparameters (see `learning/hp.py` for full list):

- Replay buffer size: 1,000,000
- Batch size: 1024
- Discount factor (γ): 0.95
- Learning rates: 1e-3 (both actor and critic)
- Soft update (τ): 0.2
- N-step: 1
- Distribution atoms: 51

