import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('result.log', 'w')
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.board[2, 0] = 1
        self.board[0, 3] = 2
        self.board[1, 3] = -2
        self.state = state
        self.is_end = False
        self.determine = DETERMINISTIC

    def get_reward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def is_end_func(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.is_end = True

    def next_position(self, action):
        if self.determine:
            if action == 'up':
                next_state = (self.state[0] - 1, self.state[1])
            elif action == 'down':
                next_state = (self.state[0] + 1, self.state[1])
            elif action == 'left':
                next_state = (self.state[0], self.state[1] - 1)
            else:
                next_state = (self.state[0], self.state[1] + 1)

            if next_state[0] >= 0 and \
                    next_state[0] <= (BOARD_ROWS - 1) and \
                    next_state[1] >= 0 and \
                    next_state[1] <= (BOARD_COLS - 1) and \
                    next_state != (1, 1):
                return next_state

            return self.state

    def show_board(self):
        self.board[self.state] = 5
        for i in range(0, BOARD_ROWS):
            logger.info('-' * 17)
            out = '|'
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 2:
                    token = ' * '
                if self.board[i, j] == -2:
                    token = ' # '
                if self.board[i, j] == 0:
                    token = ' 0 '
                if self.board[i, j] == 1:
                    token = ' s '
                if self.board[i, j] == 5:
                    token = ' x '

                out += token + '|'
            logger.info(out)
        logger.info('-' * 17)


class Agent:
    def __init__(self):
        self.states = []
        self.actions = ['up', 'down', 'left', 'right']
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0

    def choose_action(self):
        max_next_reward = 0
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                next_reward = self.state_values[self.State.next_position(a)]
                if next_reward >= max_next_reward:
                    action = a
                    max_next_reward = next_reward
        return action

    def take_action(self, action):
        position = self.State.next_position(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:

            if self.State.is_end:

                reward = self.State.get_reward()
                self.state_values[self.State.state] = reward
                logger.info(f'Game end reward={reward}')
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.choose_action()
                self.states.append(self.State.next_position(action))
                logger.info(f"Current position is {self.State.state}, action is {action}")

                self.State.show_board()
                self.State = self.take_action(action)
                self.State.is_end_func()
                logger.info(f"Next state is {self.State.state}")

                self.State.show_board()

            self.show_values()
            logger.info('\n')

    def show_values(self):
        for i in range(0, BOARD_ROWS):
            logger.info('-' * 29)
            out = '|'
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + '|'
            logger.info(out)
        logger.info('-' * 29)


if __name__ == '__main__':
    agent = Agent()
    agent.State.show_board()
    agent.play(50)
    # agent.show_values()
