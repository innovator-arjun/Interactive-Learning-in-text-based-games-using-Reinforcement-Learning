import torch
from torch import nn
from torch.nn import functional as F


class MemoryNet(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", keys=3,
                 write_threshold=0, max_mem_size=30, use_last_action=False,
                 neighbors=False, graph=False):
        super(MemoryNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.controller_gru = nn.GRU(hidden_size, hidden_size)

        self.writer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
        self.reader = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, keys * hidden_size))
        input_size = hidden_size + keys * hidden_size
        if graph:
            input_size += 4 * keys * hidden_size
        elif neighbors:
            input_size += 2 * keys * hidden_size
        self.dqn = nn.Sequential(nn.Linear(input_size, int(input_size * .75)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .75), int(input_size * .5)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .5), int(input_size * .25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .25), 1))

        self.controller_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.states = None
        self.last_cmds = None
        self.last_action = None
        self.prev_states = None
        self.next_states = None
        self.last_state = None

        self.keys = keys
        self.hidden_size = hidden_size
        self.device = device
        self.write_threshold = write_threshold
        self.max_mem_size = max_mem_size
        self.use_last_action = use_last_action
        self.neighbors = neighbors
        self.graph = graph

    def forward(self, obs, commands):  # Batch size must be 1
        nb_cmds = commands.size(1)

        # Get state and command encodings
        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)  # len x batch x hidden
        obs_encoder_hidden = self.obs_encoder_gru(obs_embedding)[1]  # 1 x batch x hidden
        cmds_embedding = self.embedding(commands)  # len x cmds x hidden
        cmds_encoder_hidden = self.cmd_encoder_gru(cmds_embedding)[1]  # 1 x cmds x hidden

        controller_out, controller_hidden = self.controller_gru(obs_encoder_hidden, self.controller_hidden)  # 1 x batch x hidden
        self.controller_hidden = controller_hidden

        # Update model
        if self.states is None:
            self.states = obs_encoder_hidden.squeeze(0)
            if self.graph:
                self.prev_states = torch.zeros((1, 2 * self.hidden_size), device=self.device)
                self.next_states = torch.zeros((1, 2 * self.hidden_size), device=self.device)
                self.last_state = 0
            elif self.neighbors:
                self.prev_states = torch.tensor([-1], device=self.device, dtype=torch.long)
                self.next_states = torch.tensor([-1], device=self.device, dtype=torch.long)
                self.last_state = 0

        else:
            key = self.writer(controller_out).squeeze(0)  # 1 x hidden
            similarities = F.cosine_similarity(self.states, key)  # N
            mx, argmx = torch.max(similarities, 0)

            if mx > self.write_threshold or self.states.shape[0] > self.max_mem_size:
                state_vector = torch.zeros_like(self.states)
                state_vector[argmx] = obs_encoder_hidden
                state_mask = torch.ones_like(self.states)
                state_mask[argmx] = torch.zeros_like(self.states[0])
                self.states = torch.mul(self.states, state_mask)
                self.states = self.states + state_vector

                if self.graph or self.neighbors:
                    state_vector = torch.zeros_like(self.prev_states)
                    if self.graph:
                        state_vector[argmx] = torch.cat((self.states[self.last_state], self.last_cmds[self.last_action].squeeze(0)), dim=-1)
                    else:
                        state_vector[argmx] = self.last_state
                    state_mask = torch.ones_like(self.prev_states)
                    state_mask[argmx] = torch.zeros_like(self.prev_states[0])
                    self.prev_states = torch.mul(self.prev_states, state_mask)
                    self.prev_states = self.prev_states + state_vector

                    state_vector = torch.zeros_like(self.next_states)
                    if self.graph:
                        state_vector[self.last_state] = torch.cat((self.states[argmx], self.last_cmds[self.last_action].squeeze(0)), dim=-1)
                    else:
                        state_vector[self.last_state] = argmx
                    state_mask = torch.ones_like(self.next_states)
                    state_mask[self.last_state] = torch.zeros_like(self.next_states[0])
                    self.next_states = torch.mul(self.next_states, state_mask)
                    self.next_states = self.next_states + state_vector

                    self.last_state = argmx.item()

            else:
                new_state = obs_encoder_hidden.squeeze(0)
                self.states = torch.cat((self.states, new_state), dim=0)
                
                if self.graph or self.neighbors:
                    if self.graph:
                        new_state = torch.cat((self.states[self.last_state], self.last_cmds[self.last_action].squeeze(0)), dim=-1).unsqueeze(0)
                    else:
                        new_state = torch.tensor([self.last_state], device=self.device, dtype=torch.long)
                    self.prev_states = torch.cat((self.prev_states, new_state), dim=0)

                    state_vector = torch.zeros_like(self.next_states)
                    if self.graph:
                        state_vector[self.last_state] = torch.cat((self.states[-1], self.last_cmds[self.last_action].squeeze(0)), dim=-1)
                    else:
                        state_vector[self.last_state] = self.states.shape[0] - 1
                    state_mask = torch.ones_like(self.next_states)
                    state_mask[self.last_state] = torch.zeros_like(self.next_states[0])
                    self.next_states = torch.mul(self.next_states, state_mask)
                    self.next_states = self.next_states + state_vector
                    self.next_states = torch.cat((self.next_states, torch.tensor([-1], device=self.device, dtype=torch.long)), dim=0)
                    
                    self.last_state = self.states.shape[0] - 1

        # Get keys and read strength
        keys = self.reader(controller_out.squeeze(0)).reshape(self.keys, self.hidden_size)  # keys x hidden

        # Get weights
        similarities = []
        for i in range(keys.shape[0]):
            similarities.append(F.cosine_similarity(self.states, keys[i].unsqueeze(0)))
        similarities = torch.stack(similarities)  # keys x memory
        weights = F.softmax(similarities, dim=1)

        # Get states and actions
        state_input = torch.mm(weights, self.states)  # keys x hidden
        state_input = torch.flatten(state_input)

        # Get previous states
        if self.graph:
            prev_input = torch.mm(weights, self.prev_states)
            next_input = torch.mm(weights, self.next_states)
            state_input = torch.cat((state_input, torch.flatten(prev_input), torch.flatten(next_input)), dim=-1)
        elif self.neighbors:
            prv = []
            nxt = []
            for i in range(self.states.shape[0]):
                if self.prev_states[i].item() == -1:
                    prv.append(torch.zeros_like(self.states[0]))
                else:
                    prv.append(self.states[self.prev_states[i]])
                if self.next_states[i].item() == -1:
                    nxt.append(torch.zeros_like(self.states[0]))
                else:
                    nxt.append(self.states[self.next_states[i]])
            prv = torch.stack(prv)
            nxt = torch.stack(nxt)
            prev_input = torch.mm(weights, prv)
            next_input = torch.mm(weights, nxt)
            state_input = torch.cat((state_input, torch.flatten(prev_input), torch.flatten(next_input)), dim=-1)

        # Get dqn input
        state_input = torch.stack([state_input.unsqueeze(0).unsqueeze(0)] * nb_cmds, 2)  # 1 x batch x cmds x keys*hidden
        dqn_input = torch.cat((state_input, cmds_encoder_hidden.unsqueeze(0)), dim=-1)  # 1 x batch x cmds x (keys+1)hidden

        # Compute q
        scores = self.dqn(dqn_input).squeeze(-1)  # 1 x batch x cmds

        # Store last actions
        if self.use_last_action:
            self.last_cmds = commands
        else:
            self.last_cmds = cmds_encoder_hidden.squeeze(0)

        return scores

    def reset_hidden(self, batch_size):
        self.controller_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        self.states = None
        self.last_cmds = None
        self.last_action = None
        self.prev_states = None
        self.next_states = None
        self.last_state = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        s = self.states.detach().cpu().numpy() if self.states is not None else None
        p = self.prev_states.detach().cpu().numpy() if self.prev_states is not None else None
        n = self.next_states.detach().cpu().numpy() if self.next_states is not None else None

        return [self.controller_hidden.detach().cpu().numpy(), c, s, self.last_action, p, n, self.last_state]

    def set_hidden(self, tensors):
        self.controller_hidden = torch.from_numpy(tensors[0]).to(self.device)
        self.last_cmds = torch.from_numpy(tensors[1]).to(self.device) if tensors[1] is not None else None
        self.states = torch.from_numpy(tensors[2]).to(self.device) if tensors[2] is not None else None
        self.last_action = tensors[3]
        self.prev_states = torch.from_numpy(tensors[4]).to(self.device) if tensors[4] is not None else None
        self.next_states = torch.from_numpy(tensors[5]).to(self.device) if tensors[5] is not None else None
        self.last_state = tensors[6]


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", use_last_action=False):
        super(RecurrentNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.state_gru = nn.GRU(hidden_size, hidden_size)

        input_size = 2 * hidden_size
        self.dqn = nn.Sequential(nn.Linear(input_size, int(input_size * .75)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .75), int(input_size * .5)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .5), int(input_size * .25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .25), 1))

        self.hidden_size = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.last_cmds = None
        self.last_action = None
        self.device = device
        self.use_last_action = use_last_action

    def forward(self, obs, commands):
        nb_cmds = commands.size(1)

        # Get state and command encodings
        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)
        obs_encoder_hidden = self.obs_encoder_gru(obs_embedding)[1]
        cmds_embedding = self.embedding(commands)
        cmds_encoder_hidden = self.cmd_encoder_gru(cmds_embedding)[1]  # 1 x cmds x hidden

        # Get state representation
        state_hidden = self.state_gru(obs_encoder_hidden, self.state_hidden)[1]
        self.state_hidden = state_hidden

        # Get dqn input
        state_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        dqn_input = torch.cat((state_input, cmds_encoder_hidden.unsqueeze(0)), dim=-1)  # 1 x batch x cmds x hidden

        # Compute q
        scores = self.dqn(dqn_input).squeeze(-1)  # 1 x batch x cmds

        # Store last actions
        self.last_cmds = commands

        return scores

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        self.last_cmds = None
        self.last_action = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        return [self.state_hidden.detach().cpu().numpy(), c, self.last_action]

    def set_hidden(self, tensors):
        self.state_hidden = torch.from_numpy(tensors[0]).to(self.device)
        self.last_cmds = torch.from_numpy(tensors[1]).to(self.device) if tensors[1] is not None else None
        self.last_action = tensors[2]


class NaiveNet(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", use_last_action=False):
        super(NaiveNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)

        input_size = 2 * hidden_size
        self.dqn = nn.Sequential(nn.Linear(input_size, int(input_size * .75)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .75), int(input_size * .5)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .5), int(input_size * .25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .25), 1))

        self.hidden_size = hidden_size
        self.last_cmds = None
        self.last_action = None
        self.device = device
        self.use_last_action = use_last_action

    def forward(self, obs, commands):
        nb_cmds = commands.size(1)

        # Get state and command encodings
        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)
        obs_encoder_hidden = self.obs_encoder_gru(obs_embedding)[1]
        cmds_embedding = self.embedding(commands)
        cmds_encoder_hidden = self.cmd_encoder_gru(cmds_embedding)[1]  # 1 x cmds x hidden

        # Get dqn input
        state_input = torch.stack([obs_encoder_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        dqn_input = torch.cat((state_input, cmds_encoder_hidden.unsqueeze(0)), dim=-1)  # 1 x batch x cmds x hidden

        # Compute q
        scores = self.dqn(dqn_input).squeeze(-1)  # 1 x batch x cmds

        # Store last actions
        self.last_cmds = commands

        return scores

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        self.last_cmds = None
        self.last_action = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        return [self.state_hidden.detach().cpu().numpy(), c, self.last_action]

    def set_hidden(self, tensors):
        self.state_hidden = torch.from_numpy(tensors[0]).to(self.device)
        self.last_cmds = torch.from_numpy(tensors[1]).to(self.device) if tensors[1] is not None else None
        self.last_action = tensors[2]
