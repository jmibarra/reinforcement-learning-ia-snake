import torch
import random
import numpy as np
from collections import deque
from snake_pygame import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Para controlar la eleatoriedad
        self.gamma = 0  # Tasa de descuento
        # Si superamos la memoria removemos elementos de la izquierda "popleft()"
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None  # TODO: implementar
        self.trainer = None  # TODO: implementar

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y + 20)
        point_d = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGTH
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straighy
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger rigth
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x  # food left
            game.food.x > game.head.x  # food rigth
            game.food.y < game.head.y  # food.up
            game.food.y > game.head.y  # food down

        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Si pasa el MAX_MEMORY hace un popLeft
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # Devuelve una lista de tuplas
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Agrupo el valor de cada elemento de cada tupla en un array con cada conjunto de datos
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # obtengo el antiguo estado
        state_old = agent.get_state(game)

        # obtengo el movimiento
        final_move = agent.get_action(state_old)

        # Muevo y obtengo el nuevo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Entreno la memoria corta, 1 paso.
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Entreno memoria larga, experiencia
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # TODO: agent.mode.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()
