import torch
import math
from SnakeGameForAI import Game
from agent import Agent
from timer import FPS_Clock

# Hyperparameters --------------------------------------

NUM_GAME_TO_PLAY = 50

# Calculate metrics ------------------------------------


def get_mean(score_list):

    total_score = 0
    for i in score_list:
        total_score += i
    mean_score = total_score / len(score_list)

    return mean_score


def get_median(score_list):

    score_list.sort()

    if len(score_list) % 2 == 0:

        upper = int(len(score_list) / 2)
        lower = int(len(score_list) / 2 - 1)

        median_score = (score_list[upper] + score_list[lower]) / 2

    else:

        mid = math.floor(len(score_list) / 2)
        median_score = score_list[mid]

    return median_score


# Print metrics --------------------------------------


def print_metrics(highest_score, score_list):

    # Calculate mean
    mean_score = get_mean(score_list)

    # Calculate median
    median_score = get_median(score_list)


    print("Highest score:", highest_score)
    print("Mean score:", mean_score)
    print("Median score:", median_score)

# Main -------------------------------------------------

def main():

    # Setup --------------------------------------------

    # Game counter
    cur_game_played = 0

    # Agent
    agent = Agent()
    agent.epsilon = 0
    agent.epsilon_end = 0
    for p in agent.model.parameters():
        p.requires_grad = False

    # Score
    best_score = agent.best_score
    score_list = []

    # Game
    game = Game()
    # Frames per second (FPS)
    fps_clock = FPS_Clock(game.fps)

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.model.to(device)
    agent.target_model.to(device)

    # Play loop ------------------------------------------

    game.main()
    state = agent.get_state()

    while True:

        fps_clock.start_timer()

        action = agent.get_action(state)
        game.AI_move(action)
        game.main_loop(agent)

        _, __, score = game.get_feed_back_for_agent()

        state = agent.get_state()
        fps_clock.tick()

        if game.is_game_over:

            game.game_over()
            cur_game_played += 1
            score_list.append(score)

            if best_score < score:
                best_score = score

            if cur_game_played == NUM_GAME_TO_PLAY:
                break

            else:
                game.reset_game()
                state = agent.get_state()

    # Print metrics --------------------------------------

    print_metrics(highest_score=best_score, score_list=score_list)


if __name__ == "__main__":
    main()