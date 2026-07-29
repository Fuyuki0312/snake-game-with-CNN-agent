import torch
import mss
import random
import os
from PIL import Image
from collections import deque
from torchvision import transforms
from model import CNN_QLearning
from SnakeGameForAI import canvas, window
from trainer import Q_Learning

# Hyperparameters --------------------------------------

MAX_MEMORY = 20_000
BATCH_SIZE = 128
GAMMA = 0.97
LEARNING_RATE = 0.0004
RESIZED_SCREENSHOT = (75, 75)

# EPSILON
EPSILON_START = 1.00
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999

OBSERVE_n_IMAGES_AT_THE_SAME_TIME = 4
UPDATE_MODEL_AFTER_n_ACTIONS = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# Agent ------------------------------------------------

class Agent:

    def __init__(self, file_name_to_save="CNNSnakeModel.pth"):

        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        # Counter
        self.n_games = 0
        self.n_actions = 0 # to update target_model
        self.steps_since_food = 0 # to end game when agent gets stuck in a loop
        self.n_actions_train_long_memory = 0

        # Epsilon
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

        # Best score
        self.best_score = 0

        # Model
        self.model = CNN_QLearning(in_channels=OBSERVE_n_IMAGES_AT_THE_SAME_TIME * 3)
        self.target_model = CNN_QLearning(in_channels=OBSERVE_n_IMAGES_AT_THE_SAME_TIME * 3)
        self.target_model.load_state_dict(self.model.state_dict())
        for p in self.target_model.parameters():
            p.requires_grad = False # I don't want to keep track of target_model's gradient
        self.update_target_model_after_n_actions = UPDATE_MODEL_AFTER_n_ACTIONS

        self.trainer = Q_Learning(model=self.model, target_model=self.target_model, gamma=self.gamma, lr=LEARNING_RATE)

        self.transform = transforms.Compose([
            transforms.Resize(RESIZED_SCREENSHOT),
            transforms.ToTensor()
        ])

        self.frame_buffer = deque(maxlen=OBSERVE_n_IMAGES_AT_THE_SAME_TIME)

        # Load agent's stuff from a pth file (CNNSnakeModel.pth as default)
        if os.path.exists(file_name_to_save):

            checkpoint = torch.load(f=file_name_to_save, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["model_state_dict"])
            self.memory = checkpoint["agent_memory"]
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.n_games = checkpoint["n_games"]
            self.n_actions = checkpoint["n_actions"]
            self.best_score = checkpoint["best_score"]

            for state in self.trainer.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            print("Agent has been loaded successfully")
            print(f"The agent has been trained on {self.n_games} games")
            print(f"Current best score: {self.best_score}\n")

        else: # If the file (.pth) is not found, let the coder know that

            print(f"Cannot access file {file_name_to_save}")
            print("Trying to initialize a new agent\n")


    # Save -------------------------------------------------

    def save_to_pth(self, file_name="CNNSnakeModel.pth"):

        torch.save(
            obj={
                "model_state_dict": self.model.state_dict(),
                "agent_memory": self.memory,
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "n_games": self.n_games,
                "n_actions": self.n_actions,
                "best_score": self.best_score
            },
            f=file_name
        )

    # Make state easier to store -------------------------------

    def encode_state(self, state):
        return (
            state.detach()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
        )

    def decode_state(self, state):
        return state.to(
            device=device,
            dtype=torch.float32
        ) / 255.0

    # Get state (get screenshots from the game) -----------------

    def get_state(self):

        window.update()


        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        w = canvas.winfo_width()
        h = canvas.winfo_height()

        with mss.MSS() as sct:
            bbox = {"top": y, "left": x, "width": w, "height": h}
            sct_img = sct.grab(bbox)
            image = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


        processed_image = self.transform(image)

        self.frame_buffer.append(processed_image)

        while len(self.frame_buffer) < OBSERVE_n_IMAGES_AT_THE_SAME_TIME:
            self.frame_buffer.appendleft(processed_image)

        state = torch.cat(list(self.frame_buffer), dim=0)

        return state.unsqueeze(0)


    def reset_frame_buffer(self):
        self.frame_buffer.clear()


    def add_to_memory(self, state, action, reward, next_state, is_game_over):
        self.memory.append((
            self.encode_state(state),
            action,
            reward,
            self.encode_state(next_state),
            is_game_over
        ))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, is_game_overs = zip(*mini_sample)
        states = torch.cat(states, 0) # transforms tuple(tensor) to tensor
        next_states = torch.cat(next_states, 0)

        states = self.decode_state(states)
        next_states = self.decode_state(next_states)

        self.trainer.train(states, actions, rewards, next_states, is_game_overs)


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def get_action(self, state):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.n_actions += 1
        self.n_actions_train_long_memory += 1
        self.steps_since_food += 1

        if self.epsilon > 0.05:
            print(f"Epsilon: {self.epsilon:.2f}")
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state)
            prediction_argmax = prediction.argmax().item()
            final_move[prediction_argmax] = 1

        return final_move # = [x, x, x]


    def disable_epsilon(self):
        self.epsilon = 0.0
        self.epsilon_end = 0.0
