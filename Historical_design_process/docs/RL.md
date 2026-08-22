# Reinforcement Learning (RL) Module

This module implements a Contextual Bandit approach to dynamically evaluate and weight EEG data segments during training. It helps the system dynamically decide which data should be used to train the deep learning models and how much weight they should have.

## Components

### `bandit.py`
Contains the core implementation of the contextual bandit:
- **`NeuralContextualBandit`**: A neural network-based contextual bandit (NeuralUCB). It evaluates contexts (like prediction entropy and loss) to decide actions for the corresponding EEG data segments.
- **`BanditAction`**: Defines actions that can be taken on a data segment: `INCLUDE`, `DISCARD`, `HIGH_WEIGHT`, `LOW_WEIGHT`.
- **`ContextIDX`**: Defines the indexes for the context features (e.g., `DELTA_LOSS`, `MEAN_ENTROPY`, `ALPHA_POWER_NORM`).
- **`UpdateScheduler`**: Determines when to update the bandit's neural network based on accumulated scores of actions taken.
- **`ContextNormalizer`**: Normalizes the context features incrementally.

### `RL_interface.py`
Provides an interface to connect the core game/EEG loop with the bandit model:
- Extracts context features from the current hits (e.g., entropy, alpha power).
- Maintains the state to calculate `deltaValLoss` and manages interaction with the NeuralContextualBandit.

### `RL_pretrain/`
Scripts for pre-training the RL module offline using previously collected data, simulating the closed-loop system to learn an initial policy before real-time usage.

## Workflow

1. **Context Extraction**: During gameplay or calibration, features are extracted for each data segment.
2. **Action Selection**: The bandit selects an action (`INCLUDE`, `DISCARD`, etc.) based on the context.
3. **Reward Calculation**: After deep learning model fine-tuning, the change in loss (`deltaValLoss`) is used to calculate a reward for the bandit.
4. **Update**: The bandit model is updated with the context, action, and received reward to improve future selections.
