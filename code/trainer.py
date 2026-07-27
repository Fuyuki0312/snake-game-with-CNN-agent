import torch


class Q_Learning:

    def __init__(self, model, target_model, gamma=0.9, lr=0.0003, loss_func=None, optimizer=None):

        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss() if loss_func is None else loss_func
        self.optimizer = torch.optim.Adam(params=model.parameters(),lr=lr) if optimizer is None else optimizer

        # model
        self.model = model
        self.target_model = target_model
        self.target_model.eval()

    def train(self, state, action, reward, next_state, is_game_over):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Transform to tensor and move to the device
        state = state.to(device)
        next_state = next_state.to(device)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long).to(device)
        else:
            action.to(device)

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float).to(device)
        else:
            reward.to(device)


        # Processing shape
        if isinstance(is_game_over, bool):
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            is_game_over = (is_game_over, )


        raw_pred = self.model(state)
        target = raw_pred.clone()

        with torch.inference_mode():
            next_q_values = self.target_model(next_state).max(dim=1)[0]

        if not isinstance(is_game_over, torch.Tensor):
            is_game_over_tensor = torch.tensor(is_game_over, dtype=torch.float).to(device)
        else:
            is_game_over_tensor = is_game_over.to(device)

        Q_new = reward + self.gamma * next_q_values * (1 - is_game_over_tensor)


        if action.dim() == 2 and action.shape[1] > 1:
            action_indices = action.argmax(dim=1)
        else:
            action_indices = action

        for i in range(len(is_game_over)):
            target[i][action_indices[i].item()] = Q_new[i]


        loss = self.loss_func(target, raw_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()