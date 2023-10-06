import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models

import json
import csv
import random

from environment.maze import Maze, Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level

with open('MazeList.json', 'r') as file:
    data = json.load(file)


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    RANDOM_MODEL = auto()
    Q_LEARNING = auto()
    Q_ELIGIBILITY = auto()
    SARSA = auto()
    SARSA_ELIGIBILITY = auto()
    DEEP_Q = auto()
    LOAD_DEEP_Q = auto()
    SPEED_TEST_1 = auto()
    SPEED_TEST_2 = auto()


test = Test.SARSA_ELIGIBILITY  # which test to run
maze_index = None


# if you want to remove the looped training: add this section of code and remove indentation
# maze_number = random.randint(0, 999)  # maze randomizer
# maze_number = 2  # manual selection
# print("Maze Number: ", maze_number)
#
# maze = np.array(
#     data['mazes'][maze_number]
# )  # 0 = free, 1 = occupied
#
# game = Maze(maze)

def maze_index_for_saving():
    return maze_number


for maze_number in range(23, 1000):
    print("Maze Number: ", maze_number)

    maze = np.array(
        data['mazes'][maze_number]
    )  # 0 = free, 1 = occupied

    game = Maze(maze)

    # train using tabular SARSA learning and an eligibility trace
    if test == Test.SARSA_ELIGIBILITY:
        game.render(Render.TRAINING)  # shows all moves and the q table; nice but slow.
        model = models.SarsaTableTraceModel(game)
        h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=350,
                                 stop_at_convergence=True)


        def save_optimal_actions_to_csv(model, filename):
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                maze_idx = maze_index_for_saving()

                # If the file is empty, write the header
                if file.tell() == 0:
                    writer.writerow(['MazeIndex', 'State', 'OptimalAction'])

                # Collecting all unique states from the Q table
                states = sorted(set([state for (state, _) in model.Q.keys()]), key=lambda x: (x[1], x[0]))

                for state in states:
                    optimal_action = model.predict(np.array(state).reshape(1, -1))

                    writer.writerow([maze_idx, state, optimal_action])


        save_optimal_actions_to_csv(model, "optimal_actions.csv")

    # draw graphs showing development of win rate and cumulative rewards
    try:
        h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
        fig.canvas.manager.set_window_title(model.name)
        ax1.plot(*zip(*w))
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
        ax2.plot(h)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        plt.show()
    except NameError:
        pass
    game.render(Render.MOVES)
    game.play(model, start_cell=(4, 1))

    plt.show()  # must be placed here else the image disappears immediately at the end of the program
