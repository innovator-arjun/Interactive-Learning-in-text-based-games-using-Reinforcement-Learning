import csv
import torch
import os
import sys
import getopt
import numpy as np
from textrl.agent import TextAgent
from textrl.env import TextWorldEnv
from textrl.utils import get_games
import argparse
from itertools import product
from datetime import datetime
from pathlib import Path

def textworld(config_):
    print(config)
    query_mode=True

    log_results = os.path.join(log_dir, "results", config['experiment_name']+'_'+config['model']+'_' +config['level']+ '_' + str(config['seed']) + ".csv")
    log_weights = os.path.join(log_dir, "weights", config['experiment_name']+'_'+config['model']+'_' +config['level']+ '_'+str(config['seed']) )
    
    
    env = TextWorldEnv(games, query_mode=query_mode)
    agent = TextAgent(device=device, state_type=config['model'], max_memory=max_memory)
    
    agent.train()
    print(log_results)
    with open(log_results, "w", 1) as file:
        csv_writer = csv.writer(file, delimiter=",")

        # Collect some statistics
        nb_games = 0
        avg_moves, avg_reward = [], []
        
        for i in range(0, config['steps']):
            state = env.reset()  # Start new episode.

            reward = 0
            total_reward = 0
            done = False
            nb_moves = 0
            while not done:
                command = agent.act(state, reward, done, env.admissible_commands)
                state, reward, done = env.step(command)
                nb_moves += 1
                total_reward += reward
                if nb_moves>100:
                    break
            agent.act(state, reward, done, env.admissible_commands)  # Let the agent know the game is done.
            nb_games += 1
            avg_moves.append(nb_moves)
            avg_reward.append(total_reward)
            csv_writer.writerow([nb_games,nb_moves , total_reward  ])

            print("Episodes:", nb_games, "Moves:", nb_moves, "Reward:", total_reward)

            torch.save(agent.model.state_dict(), log_weights)
            
        print(np.sum(avg_moves)/nb_games  )

    env.close()

parser = argparse.ArgumentParser(description='Dynamic arguments to run different experiments')
parser.add_argument('model', type=str,
                    help='Type of the architecture')

parser.add_argument('level', type=str,
                    help='Level of difficulty of the game')

parser.add_argument('seed', type=int,
                    help='For reproducibility and statistical inference')

args = parser.parse_args()
print(args)
best_hyperparameters = None

PARAM_GRID = list(product(
    
    ['coin_collector_experiments'], # Experiment Name 
    [args.model],  # Type of model
    [args.level], # Difficulty between 1 to 300
    [args.seed], # seed value
    [7500], # episodes
    [1], # Game count 
    
    # Mostly remains the same
    ['tw-coin_collector'], # Game type 
    ['log'], # Log directory
    ['tw_games/collector'], # Game Directory
    ['cpu'], # cpu, cuda
    [30],  # Max Memory
))

h_param_list = []

for param_ix in range(len(PARAM_GRID)):
    config = {}
    experiment_name,model, level,seed, steps,game_count,experiment, log_dir, games_dir, device, max_memory = PARAM_GRID[param_ix]

    if not os.path.exists(os.path.join(log_dir, "weights")):
        os.makedirs(os.path.join(log_dir, "weights"))
    if not os.path.exists(os.path.join(log_dir, "results")):
        os.makedirs(os.path.join(log_dir, "results"))

    games = get_games(experiment, games_dir, level, game_count=game_count, seed=seed)
 
    config['experiment_name'] = experiment_name
    config['model'] = model
    config['level'] = level
    config['experiment'] = experiment
    config['log_dir'] = log_dir
    config['games_dir'] = games_dir
    config['game_count'] = game_count
    config['device'] = device
    config['steps'] = steps
    config['max_memory'] = max_memory
    config['seed'] = seed
    config['training_games'] = games
    if config not in h_param_list:
        h_param_list.append(config)

print(len(h_param_list))
print(h_param_list)

textworld(h_param_list)