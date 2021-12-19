import random
import re
import gym
import textworld.gym
from textworld import EnvInfos, GameMaker, g_rng


class TextWorldEnv:
    def __init__(self, games, max_episode_length=100, query_mode=False):

        infos_to_request = EnvInfos(description=True, inventory=True, admissible_commands=True,
                                    won=True, lost=True, max_score=True)
        env_id = textworld.gym.register_games(games, request_infos=infos_to_request,
                                              max_episode_steps=max_episode_length)
                                              
        self.env = gym.make(env_id)
        self.last_score = 0
        self.query_mode = query_mode
        self.query_commands = []  # ["look", "inventory", "score", "goal"]
        self.admissible_commands = None
        self.info = None

    def reset(self):
        obs, info = self.env.reset()
        self.info = info
        self.last_score = 0
        self.admissible_commands = info["admissible_commands"]
        if self.query_mode:
            self.admissible_commands += self.query_commands
        random.shuffle(self.admissible_commands)
        return obs

    def step(self, action):
        obs, score, done, info = self.env.step(action)

        reward = score - self.last_score
        self.last_score = score
        self.info = info
        self.admissible_commands = info["admissible_commands"]
        if self.query_mode:
            state = obs
            self.admissible_commands += self.query_commands
        else:
            state = "{}\n{}\n{}".format(obs, info["description"], info["inventory"])

        random.shuffle(self.admissible_commands)
        return state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = TextWorldEnv()
    o = env.reset()
    for _ in range(50):
        print(o[0])
        print(env.last_score)
        a = input()
        o = env.step(a)
