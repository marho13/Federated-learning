import ray
import torch
import numpy as np
from modelFiles.PPO import PPO

class ParameterServer(object):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, numAgents, netsize):
        self.model = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std, netsize)
        self.num_agents = numAgents

    def getActorGrad(self, grad):  # My idea could decrease the number in np.divide
        avg_grad = self.getAvgGrad(*grad)#avg_grad = self.calcAvgGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, avg_grad)#avg_grad)
            output.append(temp)

        return output

    def avgWeights(self, weight):
        sumWeights = {}
        for w in weight[0]:
            sumWeights.update({w:np.zeros(list(weight[0][w].size()))})
        for w in weight:
            for x in w:
                sumWeights[x] += w[x].numpy()


        for w in weight[0]:
            sumWeights[w] = torch.from_numpy(sumWeights[w]/self.num_agents)

        self.set_weights(sumWeights)


    def getActorGradSum(self, grad):
        sum_grad = self.getSumGrad(*grad)
        output = []
        for g in grad:
            temp = np.add(g, sum_grad)
            output.append(temp)

        return output

    def rewardweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.calcAvgGrad(*output)
        return self.model.get_weights()
    
    def lossweighted(self, grad, loss):
        output = self.getLossWeightedGrads(grad, loss)
        self.calcAvgGrad(*output)
        return self.model.get_weights()

    def rewardUpscaledweighted(self, grad, rew):
        output = self.getWeightedGrads(grad, rew)
        self.upscaleGrad(*output)
        return self.model.get_weights()

    def getLossWeightedGrads(self, grad, l):
        totLoss = self.getTotRew(l)
        output = []
        for g in range(len(grad)):
            weight = (1/16) + (l[g]/totLoss)
            output.append([])
            for x in range(len(grad[g])):
                output[-1].append((grad[g][x]*weight))

        return output

    def getWeightedGrads(self, grad, rew):
        reward, miny = self.MoveRewards(rew)
        totRew = self.getTotRew(reward)
        output = []
        for g in range(len(grad)):
            if reward[g] == 0.0:
                weight = (1/16) + (1/totRew)
            else:
                weight = (1/16) + reward[g]/totRew
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
        return sum*2
    
    def upscaleGrad(self, *grad):
        summed_gradients = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*grad)
                ]
        summed_gradients = np.multiply(summed_gradients, (self.num_agents/2))
        self.updater(summed_gradients)
        return self.model.get_weights()

    def calcAvgGrad(self, *gradients):
        # [print(g) for g in zip(*gradients)]
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        # summed_gradients = np.multiply(summed_gradients, (self.num_agents/2))
        #self.updater(summed_gradients)
        summed_gradients = np.divide(summed_gradients, self.num_agents) #The averaged gradient
        self.updater(summed_gradients)
        return self.model.get_weights()

    def calcGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.updater(summed_gradients)
        return self.model.get_weights()
    
    def getSumGrad(self, *grad):
        sum_grad = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*grad)
                ]
        return sum_grad

    def getAvgGrad(self, *gradients):
        sum_grad = [
                np.stack(grad_zip).sum(axis=0)
                for grad_zip in zip(*gradients)
                ]
        return np.divide(sum_grad, self.num_agents)

    def updater(self, gradient):
        self.model.optimizer.zero_grad()
        self.model.set_gradients(gradient)
        self.model.optimizer.step()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def returnModel(self):
        return self.model
