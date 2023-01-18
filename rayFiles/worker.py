import gym
import torch
import ray
import numpy as np
from modelFiles.memory import Memory
from modelFiles.PPO import PPO
# from PPO_Solutions.nikhilSolution.model import ActorCritic
from copy import copy

@ray.remote(max_restarts=-1, max_task_retries=2)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, env_name, netsize):
        self.model = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, netsize)
        self.env = gym.make(env_name)
        self.memory = Memory()

    def gatherReplay(self):
        r, t = [], 0
        while t < 2000:
            rew, timestep = self.performNActions(1000)
            r.append(rew)
            t += timestep
        return sum(r)/len(r)

    def getLoss(self, weights):
        self.model.policy.set_weights(weights)
        a, b, c, d = self.model.getTrainingMem()
        loss = self.model.trainKepochs(a, b, c, d)
        return [copy(self.model.policy.get_gradients()), loss]

    def calcGrad(self, weights):
        self.model.set_weights()#weights)
        rew = self.performNActions(1000)
        self.model.update()
        return self.model.get_weights(), rew

    def update(self):
        return self.model.update()

    def policyOldUp(self):
        self.model.policyOldUp()

    def performNActions(self, N):
        state, info = self.env.reset()
        # print(state, info)
        state = np.ravel(state)
        totRew = 0.0
        for t in range(N):
            if len(state) == 2: state = state[0]
            state = np.ravel(state)
            prevState = torch.from_numpy(state.copy())
            # s = self.numpyToTensor(state)
            action = self.model.select_action(torch.from_numpy(state).float())
            # print(action)
            # action = torch.argmax(action, dim=-1)
            # actionTranslated = self.translateAction(action)
            state, rew, done, doney, info = self.env.step(action) #rew, done, info, _ = self.env.step(action)#.item())#actionTranslated)
            state = np.ravel(state)
            # action = torch.from_numpy(action) #state', 'action', 'next_state', 'reward
            prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.unsqueeze(torch.from_numpy(state.copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)
            self.model.buffer.rewards.append(rew)
            self.model.buffer.is_terminals.append(done)
            # self.memory.push(prevState, action, stateMem, rew)
            totRew += rew
            if done or doney:
                # print(t)
                break
                # state = self.env.reset()
        return totRew, t
    def applyGrads(self, grad):
        self.model.optimizer.zero_grad()
        self.model.set_gradients(grad)
        self.model.optimizer.step()

    def memClear(self):
        self.model.clearBuffer()

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

    def getWeights(self):
        return self.model.policy.get_weights()

    def setWeights(self, weight):
        self.model.policy.set_weights(weight)

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy())

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]
