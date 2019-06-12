# Text Based Reinforcement Learning
Agents and tests for reinforcement learning in text based games using Textworld. 

This repo supports Textworld's coin collector and treasure hunter challenges. It includes a DQN agent with Prioritized Experience Replay and three memory models: a naive model that concatenates the last action and current state, a recurrent model that utilized a GRU, and a memory model that utilizes external memory with read and write heads.

## Dependencies
Install pytorch and textworld (TextWorld is currently only available for Linux and macOS).
```
pip install torch
pip install textworld
```

## Run Experiments
Execute `main.py` with specified options in order to run experiments. The first time experiments are run the games will need to be generated. This process takes longer depending on the difficulty chosen.

| Arg | Name             | Options                         | Default           |
|-----|------------------|---------------------------------|-------------------|
| -n  | Experiment name  | String                          | test              |
| -m  | Model type       | naive, recurrent, memory        | naive             |
| -e  | Experiment type  | coin_collector, treasure_hunter | treasure_hunter   |
| -l  | Difficulty level | coin: 1-300, treasure: 1-30     | 1                 |
| -g  | Game directory   | Path                            | tw_games/treasure |
| -d  | Log directory    | Path                            | log               |
| -c  | Game count       | Int                             | 1                 |
| -a  | Device           | cpu, cuda, cuda:#               | cpu               |
| -t  | Timesteps        | Int                             | 40000             |
| -s  | Seed             | Int, None (random)              | None              |
| -x  | Maximum memory   | Int                             | 30                |

Results and model weights will be stored in the log directory. Results can be visualized using the functions in `visualize.py`.