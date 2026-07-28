# Playing Snake Game with Deep Reinforcement Learning

<p align="center">
  <img src="assets/gameplay_screenshot.jpg" alt="Snake agent gameplay" width="350">
</p>

## Abstract

In this project, I explore how a Deep Q-Learning agent can learn to play Snake directly from screen images instead of receiving handcrafted inputs such as the snake’s coordinates or food position. At each step, four consecutive RGB screenshots are resized to 64x64 and stacked into a 12-channel state, giving the agent a basic sense of movement rather than a single frozen view of the game.  

A convolutional neural network (CNN) model processes this visual state and estimates three Q-values corresponding to moving straight, turning right, and turning left. The agent is trained using an epsilon-greedy policy, experience replay, Bellman updates, and a target network. Distance-based reward shaping provides small hints when the snake moves toward or away from food, while larger rewards and penalties are assigned to eating food, collisions, and excessively long loops.  

The best observed training score so far is 33, which shows that the agent can learn meaningful visual control.


## Details

(In process)


## References

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” arXiv preprint arXiv:1312.5602, 2013. 
