# Playing Snake Game with Deep Reinforcement Learning

<p align="center">
  <img src="assets/gameplay_screenshot.jpg" alt="Snake agent gameplay" width="350">
</p>

<p align="center"> Figure 1: Screen shot from snake game </p>


## Abstract

In this project, I explore how a Deep Q-Learning agent can learn to play Snake directly from screen images instead of receiving handcrafted inputs such as the snake’s coordinates or food position. At each step, four consecutive RGB screenshots are resized to 64x64 and stacked into a 12-channel state, giving the agent a basic sense of movement rather than a single frozen view of the game.  

A convolutional neural network (CNN) model processes this visual state and estimates three Q-values corresponding to moving straight, turning right, and turning left. The agent is trained using an epsilon-greedy policy, experience replay, Bellman updates, and a target network. Distance-based reward shaping provides small hints when the snake moves toward or away from food, while larger rewards and penalties are assigned to eating food, collisions, and excessively long loops.  

The best observed training score so far is 33, which shows that the agent can learn meaningful visual control.  


## Game Environment

The Snake game runs in a **750 × 750-pixel window** divided into a **15 × 15 grid**, where each cell is **50 × 50 pixels** (Figure 1).

* Both the snake's body segments and the food occupy one grid cell.
* Each time the snake eats food, its body grows by one cell.
* The snake's head is represented by a triangle, providing the model with additional visual information about its current movement direction.
* The game ends when the snake collides with the screen boundary or with its own body.
* 

## Details

(In process)

### Reward Design

The reward function combines **distance-based guidance**, **progressive food rewards**, and **score-scaled death penalties**.

| Event                      |                Reward | Purpose                                                                                                                                                                                                                                                                                                      |
| -------------------------- | --------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Move closer to the food    |                `+0.1` | Provides a small directional signal that encourages the snake to approach the food.                                                                                                                                                                                                                          |
| Move farther from the food |                `-0.1` | Discourages movements that increase the distance between the snake and the food.                                                                                                                                                                                                                             |
| Eat food                   |  `10 + current_score` | Rewards successful food collection while giving greater value to collecting multiple food items within the same episode. As the snake grows longer and becomes harder to control, the increasing reward encourages the agent to continue pursuing food instead of avoiding the additional collision risk.    |
| Die                        | `-10 - current_score` | Makes death increasingly costly as the agent achieves a higher score. Without this scaling, a fixed death penalty could become relatively insignificant compared with the accumulated food rewards, potentially encouraging reckless strategies that collect food quickly but end in an avoidable collision. |

In formula form:

```text
Reward =
    +0.1                  if the snake moves closer to the food
    -0.1                  if the snake moves farther from the food
    10 + current_score    if the snake eats the food
   -10 - current_score    if the snake dies
```

This design aims to balance **short-term navigation guidance** with **long-term survival and food collection**.


### Model Architecture

<p align="center">
  <img src="assets/dqn_architecture.png" alt="DQN architecture" width="1000">
</p>

<p align="center"> Figure 2: Overall Deep Q-Learning Network achitecture, where screen shots from snake game is resized to 75 x 75 RGB </p>


## References

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” arXiv preprint arXiv:1312.5602, 2013. 
