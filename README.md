# Teaching AI to Game using RL (TAG)

A research project comparing **DQN-family reinforcement learning algorithms** trained to play a game from scratch. Three architectures Dueling DQN, Double DQN (DDQN), and Dueling DDQN are implemented, trained, and evaluated, with full statistics and pre-trained models included.

---

## Algorithms

| Algorithm | Key Idea |
|-----------|----------|
| **Dueling DQN** | Splits Q(s,a) into V(s) + A(s,a); better value estimation in dense action spaces |
| **DDQN (Double DQN)** | Decouples action selection from Q-value evaluation to reduce overestimation bias |
| **Dueling DDQN** | Combines both improvements for more stable, accurate learning |

Each algorithm has a **Save** and **Load** variant — train from scratch or continue from a checkpoint.

---

## Repo Structure

```
Code/
  ddqn                     - Save.py # Train DDQN from scratch
  ddqn                     - Load.py # Resume DDQN training from checkpoint
  dueling_dqn              - Save.py
  dueling_dqn              - Load.py
  dueling_ddqn             - Save.py
  dueling_ddqn             - Load.py
models/                    # Saved model checkpoints
statistics/                # Training logs and reward curves
Videos/                    # Agent gameplay recordings across training stages
Stats.xlsx                 # Aggregated training statistics for cross-algorithm comparison
```

---

## Running

```bash
pip install -r requirements.txt

# Train DDQN from scratch
python "Code/ddqn - Save.py"

# Resume from checkpoint
python "Code/ddqn - Load.py"
```

Swap `ddqn` for `dueling_dqn` or `dueling_ddqn` to run the other variants.

---

## Tech

- **Language:** Python
- **Libraries:** TensorFlow / Keras, NumPy, OpenAI Gym
- **Tracking:** Custom stats logging → `Stats.xlsx` + matplotlib

---

## Context

Built as part of academic research into how different RL architectures learn game playing behaviour from scratch, no prior knowledge, just reward signals and time.

