import csv
import torch
import os
import sys
import getopt

from textrl.agent import TextAgent
from textrl.env import TextWorldEnv
from textrl.utils import get_games


def textworld(name, games, log_dir='log', test_games=None, max_steps=300000,
              state_type="recurrent", query_mode=True, device='cuda:0',
              max_memory=30):
    testing_freq = 10
    log_results = os.path.join(log_dir, "results", name + ".csv")
    log_weights = os.path.join(log_dir, "weights", name)

    testing_env = None
    if test_games:
        testing_env = TextWorldEnv(test_games, query_mode=query_mode)

    env = TextWorldEnv(games, query_mode=query_mode)
    agent = TextAgent(device=device, state_type=state_type, max_memory=max_memory)
    agent.train()

    with open(log_results, "w", 1) as file:
        csv_writer = csv.writer(file, delimiter=",")

        # Collect some statistics: nb_steps, final reward.
        nb_steps = 0
        testing_total_reward = 0
        testing_moves = 0
        nb_games = 0
        testing_score = 0
        avg_moves, avg_reward = [], []
        while nb_steps < max_steps:
            state = env.reset()  # Start new episode.

            reward = 0
            total_reward = 0
            done = False
            nb_moves = 0
            while not done:
                command = agent.act(state, reward, done, env.admissible_commands)
                state, reward, done = env.step(command)
                nb_moves += 1
                nb_steps += 1
                total_reward += reward
            agent.act(state, reward, done, env.admissible_commands)  # Let the agent know the game is done.

            if nb_games % testing_freq == 0 and testing_env:
                agent.test()
                testing_total_reward = 0
                testing_moves = 0
                testing_score = 0
                for _ in range(len(test_games)):
                    testing_reward = 0 
                    testing_done = False
                    state = testing_env.reset()
                    while not testing_done:
                        command = agent.act(state, testing_reward, testing_done, testing_env.admissible_commands)
                        state, testing_reward, testing_done = testing_env.step(command)
                        testing_moves += 1
                        testing_total_reward += testing_reward
                    testing_score += env.last_score
                testing_moves /= len(test_games)
                testing_total_reward /= len(test_games)
                testing_score /= len(test_games)
                agent.train()

            nb_games += 1
            avg_moves.append(nb_moves)
            avg_reward.append(total_reward)
            csv_writer.writerow([nb_steps, total_reward, nb_moves, testing_total_reward, testing_moves, env.last_score, testing_score])
            print("Steps:", nb_steps, "Moves:", nb_moves, "Reward:", total_reward, "Testing Moves:", nb_moves, "Testing Reward:", total_reward)
            torch.save(agent.model.state_dict(), log_weights)

    env.close()
    if test_games:
        testing_env.close()
    print("Moves:", avg_moves, "Reward:", avg_reward)


if __name__ == "__main__":
    experiment_name = "test"
    model = "naive"
    experiment = "tw-treasure"
    log_dir = "log"
    games_dir = "tw_games/treasure"
    level = "1"
    game_count = 1
    device = "cpu"
    steps = 500000
    max_memory = 30
    seed = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:m:e:l:g:d:c:a:t:s:x:")
    except getopt.GetoptError:
        print("Invalid args")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-n':
            experiment_name = arg
            print(experiment_name)
        elif opt in "-m":
            model = arg  # recurrent, memory, or naive
            print(model)
        elif opt in "-e":
            experiment = "tw-" + arg  # tw-treasure_hunter or tw-coin_collector
            print(experiment)
        elif opt in "-d":
            log_dir = arg
            print(log_dir)
        elif opt in "-g":
            games_dir = arg
            print(games_dir)
        elif opt in "-l":
            level = arg  # treasure: 1-30, coin: 1-300
            print(level)
        elif opt in "-c":
            game_count = int(arg)
            print(game_count)
        elif opt in "-a":
            device = arg
            print(device)
        elif opt in "-t":
            steps = int(arg)
            print(steps)
        elif opt in "-s":
            seed = int(arg)
            print(seed)
        elif opt in "-x":
            max_memory = int(arg)
            print(max_memory)

    if not os.path.exists(os.path.join(log_dir, "weights")):
        os.makedirs(os.path.join(log_dir, "weights"))
    if not os.path.exists(os.path.join(log_dir, "results")):
        os.makedirs(os.path.join(log_dir, "results"))

    games = get_games(experiment, games_dir, level, game_count=game_count, seed=seed)
    test_count = int(game_count * .05)
    training_games = games[:-test_count] if test_count > 0 else games
    testing_games = games[-test_count:] if test_count > 0 else None
    textworld(experiment_name, training_games, log_dir=log_dir,
                test_games=testing_games, state_type=model, 
                max_memory=max_memory, device=device, max_steps=steps)
