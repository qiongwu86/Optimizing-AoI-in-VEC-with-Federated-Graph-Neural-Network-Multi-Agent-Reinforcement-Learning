import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        "每个agent都有一个自己的状态"
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)  # 只需要观察自己的状态
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.type = args.type
        if args.type == 0:
            self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)  # 所有的agent的状态和动作
        elif args.type == 1:
            self.fc1 = nn.Linear(args.obs_shape[agent_id] + args.action_shape[agent_id], 64)  # 所有的agent的状态和动作
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        if self.type == 0:
            state = torch.cat(state, dim=1)
            for i in range(len(action)):
                action[i] /= self.max_action
            action = torch.cat(action, dim=1)
        else:
            action = action[0]
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
