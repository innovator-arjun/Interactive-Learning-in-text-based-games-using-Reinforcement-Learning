import csv
import torch
import re

from textrl.agent import TextAgent
from textrl.env import TextWorldEnv
from textrl.utils import get_games


def test_textworld(name, games, weights, model_state='recurrent', query_mode=False):

    env = TextWorldEnv(games, query_mode=query_mode)
    agent = TextAgent(device="cuda", state_type=model_state)
    agent.model.load_state_dict(torch.load(weights))
    agent.test()

    for i in range(len(games)):
        with open("game_texts/" + name + str(i) + ".txt", mode="a") as file:

            state = env.reset()  # Start new game
            file.write(state)
            reward, done = 0, False
            while not done:
                command = agent.act(state, reward, done, env.admissible_commands)
                state, reward, done = env.step(command)
                file.writelines(["> " + command + "\n", state])
            agent.act(state, reward, done, env.admissible_commands)  # Let the agent know the game is done.


if __name__ == "__main__":
    games = get_games("tw-treasure_hunter", "tw_games/treasure", "15", game_count=5)
    test_graph(games, "log/weights/May9_treasure_100updated_graph")
