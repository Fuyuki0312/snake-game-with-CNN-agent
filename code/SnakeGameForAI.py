# Python Snake game
# (can be run alone to play with FPS in hyperparameters section)
# ------------------------------------------------------------
import time
import numpy as np
import tkinter as tk
import random
import math
from timer import FPS_Clock

# Hyperparameters --------------------------------------------

## For game UI
GAME_WIDTH = 750
GAME_HEIGHT = 750
SPACE_SIZE = 50
BODY_PARTS = 3 # The initial length of the snake
SNAKE_COLOR = "#00FF00"
FOOD_COLOR = "#FF0000"
BACKGROUND_COLOR = "#000000"

## For agent
INIT_DIRECTION = "down"
POSITIVE_REWARD = 10.0
NEGATIVE_REWARD = -10.0
DISTANCE_BASED_REWARD = 0.1 # agent receives this reward every action without getting food
MAX_STEP = GAME_WIDTH/SPACE_SIZE * GAME_HEIGHT/SPACE_SIZE

## For visualize
FPS = 90
WAIT_TIME_PER_GAME = 1.5 # seconds # only for running this file alone

# Defualt setup ---------------------------------------------

score = 0

window = tk.Tk()
window.title("Snake game")
window.resizable(False, False)

canvas = tk.Canvas(
    window,
    bg=BACKGROUND_COLOR,
    height=GAME_HEIGHT,
    width=GAME_WIDTH
)

label = tk.Label(
    window,
    text="Score:{}".format(score),
    font=('consolas', 40)
)

# Snake ----------------------------------------------------

class Snake:

    def __init__(self):
        self.body_size = BODY_PARTS
        self.coordinates = []
        self.squares = []

        init_position_x = (
            int((GAME_WIDTH / SPACE_SIZE) / 2) * SPACE_SIZE
        )
        init_position_y = (
            int((GAME_HEIGHT / SPACE_SIZE) / 2) * SPACE_SIZE
        )

        for i in range(BODY_PARTS):
            self.coordinates.append(
                (
                    init_position_x,
                    init_position_y - i * SPACE_SIZE
                )
            )

        for index, (x, y) in enumerate(self.coordinates):

            if index == 0:
                body_part = self.create_snake_head(
                    init_position_x,
                    init_position_y,
                    INIT_DIRECTION
                )

            else:
                body_part = canvas.create_rectangle(
                    x, y,
                    x + SPACE_SIZE,
                    y + SPACE_SIZE,
                    fill=SNAKE_COLOR,
                    tag="snake"
                )

            self.squares.append(body_part)


    def create_snake_head(self, x, y, direction):
        size = SPACE_SIZE
        margin = size * 0.1

        if direction == "up":
            points = [
                x + size / 2, y + margin,
                x + size - margin, y + size - margin,
                x + margin, y + size - margin
            ]

        elif direction == "down":
            points = [
                x + margin, y + margin,
                x + size - margin, y + margin,
                x + size / 2, y + size - margin
            ]

        elif direction == "left":
            points = [
                x + margin, y + size / 2,
                x + size - margin, y + margin,
                x + size - margin, y + size - margin
            ]

        else:  # right
            points = [
                x + margin, y + margin,
                x + size - margin, y + size / 2,
                x + margin, y + size - margin
            ]

        return canvas.create_polygon(
            points,
            fill=SNAKE_COLOR,
            outline="white",
            width=2,
            tag="snake"
        )


# Food -----------------------------------------------------


class Food:

    def __init__(self, snake_coordinates, food_old_coordinates=None):
        self.coordinates = None

        while (
            self.coordinates is None or
            tuple(self.coordinates) in snake_coordinates or
            self.coordinates == food_old_coordinates
        ):

            x = random.randint(
                0,
                int(GAME_WIDTH / SPACE_SIZE) - 1
            ) * SPACE_SIZE

            y = random.randint(
                0,
                int(GAME_HEIGHT / SPACE_SIZE) - 1
            ) * SPACE_SIZE

            self.coordinates = [x, y]

        canvas.create_oval(
            x, y,
            x + SPACE_SIZE,
            y + SPACE_SIZE,
            fill=FOOD_COLOR,
            tag="food"
        )


# Game ----------------------------------------------------

class Game:

    def __init__(self):

        self.snake = Snake()
        self.food = Food(self.snake.coordinates)

        self.direction = INIT_DIRECTION
        self.is_game_over = False
        self.cur_distance_from_food = math.sqrt(
            (self.snake.coordinates[0][0] - self.food.coordinates[0])**2 +
            (self.snake.coordinates[0][1] - self.food.coordinates[1])**2
        )

        self.reward = 0
        self.max_step = MAX_STEP
        self.fps = FPS
        self.wait_time_per_game = WAIT_TIME_PER_GAME

    # Update next frame ------------------------------------------

    def next_turn(self, agent=None):

        global score

        old_head_x, old_head_y = self.snake.coordinates[0]

        x = old_head_x
        y = old_head_y

        if self.direction == "up":
            y -= SPACE_SIZE
        elif self.direction == "down":
            y += SPACE_SIZE
        elif self.direction == "left":
            x -= SPACE_SIZE
        elif self.direction == "right":
            x += SPACE_SIZE

        old_distance_from_food = self.cur_distance_from_food
        self.snake.coordinates.insert(0, (x, y))
        self.cur_distance_from_food = math.sqrt((self.snake.coordinates[0][0] - self.food.coordinates[0]) ** 2 + (self.snake.coordinates[0][1] - self.food.coordinates[1]) ** 2)


        canvas.delete(self.snake.squares[0])

        old_head_as_body = canvas.create_rectangle(
            old_head_x,
            old_head_y,
            old_head_x + SPACE_SIZE,
            old_head_y + SPACE_SIZE,
            fill=SNAKE_COLOR,
            tag="snake"
        )

        self.snake.squares[0] = old_head_as_body

        new_head = self.snake.create_snake_head(
            x, y,
            self.direction
        )

        self.snake.squares.insert(0, new_head)

        reward = (
            DISTANCE_BASED_REWARD

            if self.cur_distance_from_food < old_distance_from_food
            else -DISTANCE_BASED_REWARD
        ) # Reward will be positive if the snake comes close to food or will be negative otherwise

        if x == self.food.coordinates[0] and y == self.food.coordinates[1]:

            reward = POSITIVE_REWARD + score
            score += 1

            if not (agent is None):
                agent.steps_since_food = 0

            label.config(text="Score:{}".format(score))
            canvas.delete("food")

            food_old_coordinates = self.food.coordinates
            self.food = Food(self.snake.coordinates, food_old_coordinates)


        else:

            del self.snake.coordinates[-1]
            canvas.delete(self.snake.squares[-1])
            del self.snake.squares[-1]


        if not (agent is None):
            if agent.steps_since_food > self.max_step:
                agent.steps_since_food = 0
                self.is_game_over = True
                reward = NEGATIVE_REWARD - score


        if self.check_collisions():
            if not (agent is None):
                agent.steps_since_food = 0
            self.is_game_over = True
            reward = NEGATIVE_REWARD - score


        self.reward = reward
        window.update()


    # Get direction ----------------------------------------------

    def change_direction(self, new_direction):

        if new_direction == 'left' and self.direction != 'right':
            self.direction = new_direction

        elif new_direction == 'right' and self.direction != 'left':
            self.direction = new_direction

        elif new_direction == 'up' and self.direction != 'down':
            self.direction = new_direction

        elif new_direction == 'down' and self.direction != 'up':
            self.direction = new_direction

    def AI_move(self, action):

        # action = [straight, right, left]

        list_of_action = ["right", "down", "left", "up"]
        idx = list_of_action.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = list_of_action[idx]
        elif np.array_equal(action, [0, 1, 0]):
            idx = (idx + 1) % 4
            new_direction = list_of_action[idx]
        else:
            idx = (idx - 1) % 4
            new_direction = list_of_action[idx]

        self.direction = new_direction


    def check_collisions(self):

        x, y = self.snake.coordinates[0]

        if x < 0 or x >= GAME_WIDTH:
            return True

        elif y < 0 or y >= GAME_HEIGHT:
            return True

        for body_part in self.snake.coordinates[1:]:

            if x == body_part[0] and y == body_part[1]:
                return True

        return False


    # Game over ------------------------------------------------

    def game_over(self):

        canvas.delete(tk.ALL)

        canvas.create_text(
            canvas.winfo_width() / 2,
            canvas.winfo_height() / 2,
            font=('consolas', 70),
            text="GAME OVER",
            fill="red",
            tag="gameover"
        )
        window.update()

    def reset_game(self):
        # Reset game
        global score
        score = 0
        label.config(text="Score:{}".format(score))

        canvas.delete(tk.ALL)

        self.snake = Snake()
        self.food = Food(self.snake.coordinates)
        self.direction = "down"
        self.is_game_over = False
        self.cur_distance_from_food = math.sqrt(
            (self.snake.coordinates[0][0] - self.food.coordinates[0]) ** 2 +
            (self.snake.coordinates[0][1] - self.food.coordinates[1]) ** 2
        )

    # Game setup (needs to be called before the main loop) ------

    def main(self):

        label.pack()
        canvas.pack()

        window.update()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))

        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # This is to enable this file to be run in order to play snake game as normal
        # You can use arrow keys to control your snake
        window.bind('<Left>', lambda event: self.change_direction('left'))
        window.bind('<Right>', lambda event: self.change_direction('right'))
        window.bind('<Up>', lambda event: self.change_direction('up'))
        window.bind('<Down>', lambda event: self.change_direction('down'))


    def main_loop(self, agent=None): # Game loop
        self.next_turn(agent)
        window.update()


    def get_feed_back_for_agent(self):
        return self.reward, self.is_game_over, score


def reset_game_setup():
    fps_clock = FPS_Clock(fps=FPS)
    while True:
        game.reset_game()
        while not game.is_game_over:
            fps_clock.start_timer()
            game.next_turn()
            fps_clock.tick()
        game.game_over()
        time.sleep(game.wait_time_per_game)

if __name__ == "__main__":
    game = Game()
    game.main()
    reset_game_setup()
