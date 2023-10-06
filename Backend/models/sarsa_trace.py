import logging
import random
from datetime import datetime

import numpy as np

from environment import Status
from environment.maze import Render
from models import AbstractModel


class SarsaTableTraceModel(AbstractModel):
    """ Tabular SARSA based prediction model with eligibility trace.

        For every state (here: the agents current location ) the value for each of the actions is stored in a table.
        The key for this table is (state + action). Initially all values are 0. When playing training games
        after every move the value in the table is updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).

        To speed up learning the model keeps track of the (state, action) pairs which have been visited before and
        also updates their values based on the current reward (a.k.a. eligibility trace). With every step the amount
        in which previous values are updated decays.
    """
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.

        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="SarsaTableTraceModel", **kwargs)
        self.Q = dict()  # table with Q per (state, action) combination

    def Divide_quadrants(self):
        """ Divide the maze into four quadrants. """
        empty_cells = self.environment.empty.copy()

        # Divide the empty cells into four quadrants
        quadrant1, quadrant2, quadrant3, quadrant4 = [], [], [], []
        max_row = max(cell[0] for cell in empty_cells)
        max_col = max(cell[1] for cell in empty_cells)
        for cell in empty_cells:
            i, j = cell
            if i <= max_row // 2 and j <= max_col // 2:
                quadrant1.append((i, j))
            elif i <= max_row // 2 and j > max_col // 2:
                quadrant2.append((i, j))
            elif i > max_row // 2 and j <= max_col // 2:
                quadrant3.append((i, j))
            else:
                quadrant4.append((i, j))

        return quadrant1, quadrant2, quadrant3, quadrant4

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the model.

            :param stop_at_convergence: stop training as soon as convergence is reached

            Hyperparameters:
            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword float eligibility_decay: (lambda) eligibility trace decay rate per step (0 = no trace, 1 = no decay)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.999)
        exploration_rate = kwargs.get("exploration_rate", 0.9)
        exploration_decay = kwargs.get("exploration_decay", 0.999)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.999)
        eligibility_decay = kwargs.get("eligibility_decay", 0.3)  # 0.80 = 20% reduction
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # variables for performance reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        # training starts here
        for episode in range(1, episodes + 1):
            # optimization: make sure to start training from all possible cells
            quadrant_cycle_length = 25
            current_cycle = (episode - 1) // quadrant_cycle_length % 5
            # Get Quadrants
            quadrant1, quadrant2, quadrant3, quadrant4 = self.Divide_quadrants()

            # Select the quadrant based on the current episode number
            if current_cycle == 0:
                start_list = quadrant4.copy()
                print("Q4")
            elif current_cycle == 1:
                quadrant2 = quadrant2
                start_list = quadrant2.copy()
                print("Q2")
            elif current_cycle == 2:
                quadrant3 = quadrant3
                start_list = quadrant3.copy()
                print("Q3")
            elif current_cycle == 3:
                start_list = quadrant1.copy()
                print("Q1")
            elif current_cycle == 4:
                all_quadrants = quadrant1 + quadrant2 + quadrant3 + quadrant4
                start_list = all_quadrants.copy()
                print("Q ALL")

            if episode < 345:  # change when rendering happens
                self.environment.render(content=Render.ZeroVision)
            else:
                self.environment.render(content=Render.TRAINING)

            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple, so it can be used as dictionary key

            etrace = dict()

            if np.random.random() < exploration_rate:
                action = random.choice(self.environment.actions)
            else:
                action = self.predict(state)

            while True:
                try:
                    etrace[(state, action)] += 1
                except KeyError:
                    etrace[(state, action)] = 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                next_action = self.predict(next_state)

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                next_Q = self.Q.get((next_state, next_action), 0.0)

                delta = reward + discount * next_Q - self.Q[(state, action)]

                for key in etrace.keys():
                    self.Q[key] += learning_rate * delta * etrace[key]

                # decay the eligibility trace
                for key in etrace.keys():
                    etrace[key] *= (discount * eligibility_decay)

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break

                state = next_state
                action = next_action  # SARSA is on-policy: always follow the predicted action

                self.environment.render_q(self)
                # commented out for speed!!!!!!!

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ Policy: select the action with the highest value from the Q-table.
            Random choice if multiple actions have the same max value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        q = self.q(state)
        # print(self.q(state))

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
