import ray
import torch
import numpy as np
from rayParameter import Memory

@ray.remote
class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, model, num_agent, method):
        self.model = model(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.num_agent = num_agent
        self.memory = Memory(20000)
        self.distMethod = method

    def performdistMethod(self, *grad):
        self.distMethod(*grad)

    def normalDistLearningMethod(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        return summed_gradients

    def get_weights(self):
        return self.model.get_weights()

    def calcAvgGrad(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        summed_gradients = np.divide(summed_gradients, self.num_agent) #The averaged gradient
        return summed_gradients

    def getActorGrad(self, grad):               #My idea could decrease the number in np.divide
        avg_grad = self.calcAvgGrad(grad)
        output = []
        for g in grad:
            temp = np.add(g, avg_grad)
            output.append(temp)

        return np.array(output)

    def weightedGrad(self, loss, grad):
        weights = self.weightCalc(loss)
        weightedGrad = weights * grad
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(weightedGrad)
        ]
        return summed_gradients

    def weightCalc(self, loss):
        weights = []
        for a in range(len(loss)):
            weights.append((1/(loss[a])))
        return weights

    def getHighestRew(self, rew):
        maxy = rew[0].item()
        currentLargest = 0
        for a in range(len(rew)):
            if rew[a].item() > maxy:
                maxy = rew[a].item()
                currentLargest = a
        return currentLargest

    def gradHighestReward(self, grad, rew):     #Goodwins idea
        assert len(grad) == len(rew)
        currentLargest = self.getHighestRew(rew)
        return grad[currentLargest]

    def memHighestRew(self, mem, rew):          #Highest reward -> added to the memory of each agent
        assert len(mem) == len(rew)
        currentLargest = self.getHighestRew(rew)
        for b in range(len(mem)):
            mem[b].push(mem[currentLargest])
        return mem

    def addMem(self, memories):                 #Deepminds idea
        for mem in memories: #Each agent
            for m in mem: #Each experience
                self.memory.push(m)

    def getMem(self):                           #Connected to addMem, used for giving the memory to each agent
        return self.memory                      #Might need get batch

    def getMemBatch(self, batch_size, i):
        return self.memory[i*batch_size:(i+1)*batch_size]





