## How to use these files

All Python files should remain in the same directory because the project imports them as local modules.

### File Overview

| File                | Description                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent.py`          | Defines the `Agent` class and manages the main agent logic. It captures the visible game screen, resizes each screenshot to `75 × 75`, stacks four consecutive RGB frames into a `12 × 75 × 75` state, selects actions using an epsilon-greedy policy, stores transitions in replay memory, and handles checkpoint loading and saving. It also maintains both the online network and target network. |
| `model.py`          | Defines the `CNN_QLearning` neural network. The CNN processes the four stacked screenshots and outputs three Q-values corresponding to **move straight**, **turn right**, and **turn left**.                                                                                                                                                                                                         |
| `trainer.py`        | Defines the `Q_Learning` trainer. It calculates Bellman targets using the target network and updates the online network by minimizing Mean Squared Error with the Adam optimizer.                                                                                                                                                                                                                    |
| `train.py`          | The main training entry point. It connects the agent, game environment, replay-memory training, target-network updates, and checkpoint-saving logic. Run this file to train the agent.                                                                                                                                                                                                               |
| `SnakeGameForAI.py` | Implements the Snake game using Tkinter, including the snake, food generation, movement, collision detection, rewards, game reset logic, and keyboard controls. It can also be run independently to play the game manually.                                                                                                                                                                          |
| `timer.py`          | Defines the `FPS_Clock` class. It delays frames when necessary so that the game does not run faster than the configured FPS. It cannot make a slow frame run faster if the agent requires too much computation.                                                                                                                                                                                      |
| `non_train_play.py` | Runs the trained agent without exploration or model updates. It plays a fixed number of evaluation games and reports the highest, mean, and median scores, together with the complete score history.                                                                                                                                                                                                 |

> `agent.py`, `model.py`, `trainer.py`, and `timer.py` are supporting modules and do not need to be run directly.


## Installation

This project was developed and tested with the following package versions:

| Package     | Version        |
| ----------- | -------------- |
| PyTorch     | `2.5.1+cu121`  |
| Torchvision | `0.20.1+cu121` |
| MSS         | `10.2.0`       |
| Pillow      | `12.3.0`       |
| NumPy       | `2.3.5`        |

Install all required dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

The `+cu121` suffix indicates that PyTorch and Torchvision use builds compiled for CUDA 12.1. The included `requirements.txt` automatically adds the official PyTorch CUDA 12.1 package index.


## Training the Agent

Run the following command from the project directory:

```bash
python train.py
```

When training starts, the program will:

1. Load `CNNSnakeModel.pth` if an existing checkpoint is found.
2. Otherwise, initialize a new agent with randomly initialized model parameters.
3. Capture four consecutive screenshots of the game as each state.
4. Select actions, store transitions in replay memory, and train the online network.
5. Periodically synchronize the target network with the online network.
6. Save training progress to `CNNSnakeModel.pth` according to the checkpoint strategy defined in `train.py`.

To stop training, close the Snake game window or press `Ctrl+C` in the terminal.

> **Important:** Closing the window stops the program but does not force an immediate save. In the current implementation, the first automatic checkpoint is saved after 300 games. After that, another checkpoint is saved only when the agent achieves a new best score.

### Screen-Capture Requirement

The agent observes the game through screenshots of the visible game canvas rather than receiving coordinates directly from the environment.

During training or evaluation:

* Keep the complete game window visible.
* Do not minimize the game.
* Do not place another application over the game canvas.
* Do not lock the screen.

Anything covering the canvas may appear in the agent’s input and cause incorrect actions.

## Evaluating a Trained Agent

Place the trained checkpoint at:

```text
CNNSnakeModel.pth
```

Then run:

```bash
python non_train_play.py
```

By default, the agent plays 50 games with epsilon disabled, meaning that its actions are selected entirely from the CNN’s Q-value predictions. No model training is performed.

The number of evaluation games can be changed in `non_train_play.py`:

```python
NUM_GAME_TO_PLAY = 50
```

After evaluation, the program prints:

* Highest score
* Mean score
* Median score
* Complete score history

## Playing the Game Manually

Run:

```bash
python SnakeGameForAI.py
```

Use the arrow keys to control the snake:

* `↑` — Move up
* `↓` — Move down
* `←` — Move left
* `→` — Move right

When the snake dies, the game displays the game-over screen and automatically starts a new game after the waiting time configured in `SnakeGameForAI.py`.
