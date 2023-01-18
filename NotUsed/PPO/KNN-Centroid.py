import os
import random
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import random
import gym
from PPO import *
from model import *

import numpy as np
from collections import deque
import ray
import wandb


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminals', 'logprobs'))


class Memory(object):

    def __init__(self, capacity=1000):
        self.cap = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def getMem(self):
        output = self.memory
        self.memory = deque([], maxlen=self.cap)
        return output

    def __len__(self):
        return len(self.memory)

class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, envName, numAgents):
        self.num_agents = numAgents
        self.model = ConvNet(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.env = gym.make(envName)
        self.memory = Memory()

    def getActorGrad(self, grad, rew):  # My idea could decrease the number in np.divide
        get_weighted_grads = self.calcAvgGrad(*self.getWeightedGrads(grad, rew))#avg_grad = self.calcAvgGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, get_weighted_grads)#avg_grad)
            output.append(temp)

        return output

    def rewardweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.calcAvgGrad(*output)
        return self.model.get_weights()

    def getWeightedGrads(self, grad, rew):
        reward, miny = self.MoveRewards(rew)
        totRew = self.getTotRew(reward)
        output = []
        for g in range(len(grad)):
            if reward[g] == 0.0:
                weight = (1/totRew)
            else:
                weight = reward[g]/totRew
            output.append([])
            for x in range(len(grad[g])):
                output[g].append((grad[g][x] * weight))
        return output

    def MoveRewards(self, rew):
        minimum = min(rew)
        output = []
        for r in rew:
            output.append(r+minimum)
        return output, minimum

    def getTotRew(self, rew):
        sum = 0
        for r in rew:
            sum += r
        return sum

    def calcAvgGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.updater(summed_gradients)
        # summed_gradients = np.divide(summed_gradients, self.num_agent) #The averaged gradient
        return summed_gradients

    def updater(self, gradient):
        self.optimizer.zero_grad()
        self.model.set_gradients(gradient)
        self.optimizer.step()

    def get_weights(self):
        return self.model.get_weights()

    def performNActions(self, N):
        state = self.env.reset()
        totRew = 0.0
        for t in range(N):
            action, dist = self.model.act(state)
            state, rew, done, info = self.env.step(action.detach().numpy())
            totRew += rew
            if done:
                break
        return totRew

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]


@ray.remote(max_restarts=-1, max_task_retries=2)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, name):
        self.model = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, ConvNet, "cpu", K_epochs=4, eps_clip=0.2)
        self.env = gym.make(name)
        self.optimizer = torch.optim.SGD(self.model.policy.parameters(), lr=lr)
        self.memory = Memory()

    def compute_gradients(self, weights):
        #self.model.policy.set_weights(weights)#Not used for actor version
        rew = self.performNActions(1000)
        loss = self.model.getLossGrad(self.memory)
        # self.memory = Memory()
        return [self.model.policy.get_gradients(), loss, rew]

    def performNActions(self, N):
        state = self.env.reset()
        totRew = 0.0
        for t in range(N):
            prevState = torch.from_numpy(state.copy())

            action, dist = self.model.policy.act(state)

            state, rew, done, info = self.env.step(action.detach().numpy())#.item())#actionTranslated)

            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state.copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)
            self.memory.push(prevState, action, stateMem, rew, done, dist)
            totRew += rew
            if done:
                break

        return totRew

    def getActionLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        return self.model.policy.act(state)

    def getValueLayer(self):
        state = self.env.reset()
        # s = self.numpyToTensor(state)
        action, dist = self.model.policy.act(state)
        # print(action)
        logprobs, stateval, distentropy, actionprobs, actionlogprobs = self.model.policy.tester(state, action)
        return logprobs, stateval, distentropy, actionprobs, actionlogprobs

    def set_grads_calc_weights(self, grad):
        self.optimizer.zero_grad()
        self.model.policy.set_gradients(grad)
        self.optimizer.step()

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]

envName = "LunarLanderContinuous-v2"
env = gym.make(envName)
numActions = env.action_space.shape[0]#4
stateDim = env.observation_space.shape[0]#96 * 96 * 3
n_latent_var = 64
lr = 0.01
betas = (0.9, 0.999)
gamma = 0.9
iterations = 3000
num_workers = 8  # Set gpu to num_workers//num_gpu

ray.init(ignore_reinit_error=True)

print("Running synchronous parameter server training.")
#
wandy = wandb.init(project="Federated-learning-PPO",
           config={
               "batch_size": 16,
               "learning_rate": lr,
               "dataset": envName,
               "model": "Weighted single actor weight",
           })

for run in range(10):
    ps = ParameterServer(stateDim, numActions, n_latent_var, lr, betas, gamma, envName, num_workers)
    workers = [DataWorker.remote(stateDim, numActions, n_latent_var, lr, betas, gamma, envName) for i in
               range(num_workers)]

    current_weights = ps.get_weights()
    print("Run {}".format(run))
    for i in range(iterations):
        gradients = ray.get([worker.compute_gradients.remote(current_weights) for worker in workers])
        grads, loss, reward = [], [], []
        for output in gradients:
            grads.append(output[0])
            loss.append(output[1].item())
            reward.append(output[2].item())

        avgLoss = sum(loss) / num_workers
        avgRew = sum(reward) / num_workers
        wandb.log({"training loss": avgLoss}, step=i)
        wandb.log({"training reward": avgRew}, step=i)


        #Next do the gradient sum
        current_grads = ps.getActorGrad(grads, reward)

        [workers[j].set_grads_calc_weights.remote(current_grads[j]) for j in range(num_workers)]
        # current_weights = ps.rewardweighted(grads, reward)



        if i%10 == 9:
            rew = ps.performNActions(1000)
            print("Epoch {}, gave reward {}".format(i, rew.item()))
            wandb.log({"testing reward": rew.item()}, step=i)
        # else:
        #     print("Epoch {}, gave reward {} and loss {}".format(i, avgRew, avgLoss))

wandy.finish()

ray.shutdown()