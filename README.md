# Text Based Reinforcement Learning
We will use Reinforcement Learning to learn optimal control policies in systems where action space is defined by sentences in natural language would allow many interesting real-world
applications such as automatic optimisation of dialogue systems. Text based games with multiple endings and rewards are a promising platform for this task, since their feedback allows us to employ reinforcement learning techniques to jointly learn text representations and control policies.

Textworld's coin collector and treasure hunter challenges. This work includes a DQN agent with Prioritized Experience Replay and three memory models: a naive model that concatenates the last action and current state, a recurrent model that utilized a GRU, and a memory model that utilizes external memory with read and write heads.

## Dependencies
Install pytorch and textworld (TextWorld is currently only available for Linux and macOS).
```
pip install torch
pip install textworld
```

## Run Experiments
Execute `main.py` with specified options in order to run experiments. The first time experiments are run the games will need to be generated. This process takes longer depending on the difficulty of the game.

Results and model weights will be stored in the log directory.