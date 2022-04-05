import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from parameterServer import ParameterServer
import ray
from rayParameter import Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step // self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate


class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        # torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = ConvNet(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas)
        self.target_net = ConvNet(state_dim, action_dim, n_latent_var)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()

        self.MseLoss = nn.MSELoss()

    def update(self, memory, BATCH_SIZE):
        for _ in range(len(memory)//BATCH_SIZE):
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            # print(batch.next_state[0].shape, batch.state[0].shape)
            # print(type(batch.next_state[0]), type(batch.next_state))
            # print(len(batch.next_state), len(batch.state))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch)
            # print(action_batch.shape, state_action_values.shape)
            state_action_values = state_action_values.gather(dim=-1, index=action_batch.view(64, 1))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)


class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        # self.device = device
        self.randPolicy = {"Rand": 0, "Policy": 0}
        self.current_step = 0
        self.num_actions = outputSize
        self.fc1 = nn.Linear(in_features=stateDim, out_features=n_latent_var).float()
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var).float()
        self.out = nn.Linear(in_features=n_latent_var, out_features=outputSize).float()

    def forward(self, t):
        t = torch.flatten(t, start_dim=1).float()
        # t = t.flatten().float()
        # print(t.shape)
        t = self.fc1(t).float()
        t = F.relu(t).float()
        t = self.fc2(t).float()
        t = F.relu(t).float()
        t = self.out(t).float()
        return t
        # return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


@ray.remote(max_restarts=-1, max_task_retries=2)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.DQN = DQN(state_dim, action_dim, n_latent_var, lr, betas, gamma)
        self.env = gym.make("CarRacing-v0")
        self.optimizer = torch.optim.SGD(self.DQN.policy_net.parameters(), lr=lr)
        self.memory = Memory()

    def gatherReplay(self):
        # print("Performing Actions in environment")
        rew = self.performNActions(1000)
        # print("Done with gym")
        return [self.memory, rew]

    def compute_gradients(self, memory):
        self.DQN.update(memory, 64)

    def performNActions(self, N):
        state = self.env.reset()
        totRew = 0.0
        for t in range(N):
            prevState = torch.from_numpy(state.copy())
            s = self.numpyToTensor(state)
            action = self.DQN.policy_net(s)
            action = torch.argmax(action, dim=-1)
            actionTranslated = self.translateAction(action)
            state, rew, done, info = self.env.step(actionTranslated)
            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state.copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            self.memory.push(prevState, action, stateMem, rew)
            totRew += rew
            if done:
                print(t)
                break
                # state = self.env.reset()
        return totRew

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]


def performNActions(DQN, N, env):
    # memory = Memory()
    state = env.reset()
    rewardList = []
    for t in range(N):
        prevState = state
        s = numpyToTensor(state)
        action = DQN.policy_net(s)
        action = translateAction(torch.argmax(action, dim=-1))
        state, rew, done, info = env.step(action)
        # memory.push(prevState, action, state, rew)
        rewardList.append(rew)
        if done:
            break

    return sum(rewardList)


def numpyToTensor(state):
    s = np.expand_dims(state, axis=0)
    s = np.swapaxes(s, 1, -1)
    return torch.from_numpy(s.copy())


def translateAction(action):
    actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                  3: np.array([1.0, 0, 0])}
    return actionDict[action.item()]


numActions = 4
stateDim = 96 * 96 * 3
n_latent_var = 64
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.9
iterations = 200
num_workers = 16  # Set gpu to num_workers//num_gpu
env = gym.make("CarRacing-v0")

ray.init(ignore_reinit_error=True)
ps = ParameterServer.remote(stateDim, numActions, n_latent_var, lr, betas, gamma)
workers = [DataWorker.remote(stateDim, numActions, n_latent_var, lr, betas, gamma) for i in range(num_workers)]

model = DQN(stateDim, numActions, n_latent_var, lr, betas, gamma)
# test_loader = get_data_loader()[1]

print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()
for i in range(iterations):
    mem = ray.get([worker.gatherReplay.remote(current_weights) for worker in workers])
    memory, rew = [], []
    for output in mem:
        memory.append(output[0])
        rew.append(output[1])

    mems = ps.addMem.remote(mem)
    [w.compute_gradients.remote(mems) for w in workers]

    print("Done running workers")
ray.shutdown()