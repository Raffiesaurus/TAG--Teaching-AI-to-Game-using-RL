# TAG — Teaching AI to Game using RL

A research project exploring **Reinforcement Learning** applied to game agents. Three DQN-family algorithms — DQN, Dueling DQN, and Double DQN (DDQN) — are trained and evaluated in a game environment, with training statistics tracked and compared across runs.

## About

TAG was developed as part of academic research into how well different RL architectures learn to play a game from scratch. The project implements and compares three variants of Deep Q-Networks:

- **DQN** — the baseline Deep Q-Network
- **Dueling DQN** — separates state value and action advantage estimation for more stable learning
- **DDQN (Double DQN)** — decouples action selection from Q-value evaluation to reduce overestimation bias
- **Dueling DDQN** — combines both improvements

Each algorithm has a save and load variant for continuing training across sessions. Training statistics are logged and stored in `Stats.xlsx` and the `statistics/` directory for analysis. Pre-trained models are included in `models/` and training progression videos are in `Videos/`.

## Repo Structure

```
Code/           # Python training scripts (save/load variants per algorithm)
models/         # Saved model checkpoints
statistics/     # Training logs and reward curves
Videos/         # Agent gameplay recordings at various training stages
Stats.xlsx      # Aggregated training statistics for comparison
```

## Algorithms Implemented

| Algorithm | Key Idea |
|---|---|
| DQN | Neural network approximates Q-values; experience replay + target network |
| Dueling DQN | Splits Q into V(s) + A(s,a); better value estimation in dense action spaces |
| Double DQN | Uses online network to select actions, target network to evaluate them |
| Dueling DDQN | Combines dueling architecture with double Q-learning |

## Tech

- **Language:** Python
- **Libraries:** TensorFlow / Keras, NumPy, OpenAI Gym (or custom environment)
- **Tracking:** Custom stats logging → Excel / matplotlib

## Running

```bash
pip install -r requirements.txt

# Train from scratch
python "Code/ddqn - Save.py"

# Continue from checkpoint
python "Code/ddqn - Load.py"
```

Swap `ddqn` for `dueling_dqn` or `dueling_ddqn` to run other variants.
