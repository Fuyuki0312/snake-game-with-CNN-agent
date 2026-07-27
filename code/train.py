import torch
import time
from SnakeGameForAI import Game
from agent import Agent
from timer import FPS_Clock

# Hyperparameters --------------------------------------

TRAIN_LONG_MEMORY_AFTER_n_ACTIONS = 4
SAVE_BASED_ON_SCORE_AFTER_n_GAMES = 300
FILE_NAME = "CNNSnakeModel.pth"

# Save -------------------------------------------------

def save(agent):

    try:
        agent.save_to_pth(file_name=FILE_NAME)
    except Exception as e:
        print(e)
        raise Exception("Something goes wrong when trying to save the model's weights")


# Train ------------------------------------------------

def train():

    # Setup --------------------------------------------

    agent = Agent(file_name_to_save=FILE_NAME)
    game = Game()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.model.to(device)
    agent.target_model.to(device)
    fps_clock = FPS_Clock(fps=game.fps)

    # Train loop ------------------------------------------

    game.main()
    old_state = agent.get_state() # Get old state

    while True:

        fps_clock.start_timer() # Ensure the game is still running with given FPS even when the agent is fast


        action = agent.get_action(old_state)
        game.AI_move(action)
        game.main_loop(agent)

        # Get feedback for agent
        reward, is_game_over, score = game.get_feed_back_for_agent()

        # Get new state
        new_state = agent.get_state()

        # Train long memory (train on the entire memory)
        if agent.n_actions_train_long_memory >= TRAIN_LONG_MEMORY_AFTER_n_ACTIONS:
            agent.n_actions_train_long_memory = 0
            agent.train_long_memory()

        # Add to memory
        agent.add_to_memory(
            old_state, action, reward, new_state, is_game_over
        )

        fps_clock.tick() # Go to the next frame

        if game.is_game_over:
            # Note: if something is unclear, take a look at SnakeGameForAI.py
            game.game_over()
            agent.n_games = agent.n_games + 1

            if agent.n_actions > agent.update_target_model_after_n_actions:
                agent.n_actions = 0
                agent.update_target_network()
                print("\n----------- Agent's target network updated -----------\n")


            # Save

            # Save strategy:
            # Save immediately when agent have played (SAVE_BASED_ON_SCORE_AFTER_n_GAMES) games
            # After that, only save when agent gets a new best_score

            is_new_best_score = score > agent.best_score

            if is_new_best_score:
                agent.best_score = score

            if (
                    agent.n_games == SAVE_BASED_ON_SCORE_AFTER_n_GAMES or
                    (
                        agent.n_games > SAVE_BASED_ON_SCORE_AFTER_n_GAMES and
                        is_new_best_score
                    )
            ):
                save(agent)
                print("\n" + time.ctime() + ":\nBest score: " + str(agent.best_score) + "\n")
                print("-------------- Model saved successfully --------------\n")

            # Saving done here


            agent.reset_frame_buffer()
            agent.train_long_memory()
            game.reset_game()

            old_state = agent.get_state()

        else:
            old_state = new_state


if __name__ == "__main__":
    train()
