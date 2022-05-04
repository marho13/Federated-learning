import os
import random
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from torchvision.transforms import Resize, InterpolationMode
from collections import namedtuple
from collections import Counter
from collections import deque
from model import Resnet20, tinyNet
import ray
import mnistReader as mnist


@ray.remote
class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.model = Resnet20(state_dim, action_dim, n_latent_var)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.reshaper = Resize((224, 224), InterpolationMode.BILINEAR)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def performNActions(self, batch, lengthy):
        batch[0] = self.reshaper(batch[0])
        # print(batch[0].shape)
        pred = self.model(batch[0])
        predArgmax = torch.argmax(pred, dim=-1)
        loss = self.criterion(pred.float(), batch[1].float())
        acc = torch.sum(torch.eq(predArgmax, torch.argmax(batch[1], dim=-1))) / lengthy
        return loss, acc

@ray.remote(max_restarts=-1, max_task_retries=2, memory=2048*1024*1024)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.model = Resnet20(state_dim, action_dim, n_latent_var).float()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.reshaper = Resize((224, 224), InterpolationMode.BILINEAR)

    def compute_gradients(self, weights, batch, lengthy):
        self.model.set_weights(weights)
        self.optimizer.zero_grad()

        loss, acc = self.performNActions(batch, lengthy)

        loss.backward()
        return [self.model.get_gradients(), loss, acc]

    def performNActions(self, batch, lengthy):
        batch[0] = self.reshaper(batch[0])
        # print(batch[0].shape)
        pred = self.model(batch[0])
        predArgmax = torch.argmax(pred, dim=-1)
        loss = self.criterion(pred.float(), batch[1].float())
        acc = torch.sum(torch.eq(predArgmax, torch.argmax(batch[1], dim=-1)))/lengthy
        return loss, acc



batchsize = 32
numActions = 10
stateDim = 1
n_latent_var = 64
lr = 0.01
betas = (0.9, 0.999)
gamma = 0.9
iterations = 200
num_workers = 8  # Set gpu to num_workers//num_gpu
env = gym.make("CarRacing-v0")

ray.init(ignore_reinit_error=True)
ps = ParameterServer.remote(stateDim, numActions, n_latent_var, lr, betas, gamma)
workers = [DataWorker.remote(stateDim, numActions, n_latent_var, lr, betas, gamma) for i in range(num_workers)]

criterion = nn.MSELoss()
model = tinyNet(stateDim, numActions, n_latent_var).float()

print("Running synchronous parameter server training.")
current_weights = ps.get_weights.remote()

x_train, y_train, x_test, y_test = mnist.getMnist()
x_train = torch.unsqueeze(x_train, 1)
x_test = torch.unsqueeze(x_test, 1)
y_train = one_hot(y_train)
y_test = one_hot(y_test)
batch = len(x_train)//(num_workers*batchsize)
print(batch, len(x_train), num_workers, batchsize)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# ray.put(x_train)
# ray.put(y_train)
# ray.put(x_test)
# ray.put(y_test)

for i in range(iterations):
    print("Epoch {}".format(i))
    grads, loss, acc = [], 0.0, 0.0
    for b in range(batch):
        gradients = ray.get([workers[j].compute_gradients.remote(current_weights, [x_train[batchsize*(b+j):batchsize*(b+j+1)], y_train[batchsize*(b+j):batchsize*(b+j+1)]], batch) for j in range(num_workers)])
        for output in gradients:
            grads.append(output[0])
            loss += output[1]
            acc += output[2]

        current_weights = ps.apply_gradients.remote(*grads)
        del gradients
        grads = []
        print("{}/{}".format(b+1, batch))
    loss /= (num_workers*batch)
    acc /= (num_workers*batch)
    print("Training accuracy: {}               Training Loss: {}".format(acc, loss))

    print("Done running workers")
    if i % 10 == 0:
        loss = 0.0
        acc = 0.0
        batSize = len(x_test)//(batchsize*num_workers)
        for x in range(batSize):
            batch_loss, batch_acc = ray.get(ps.performNActions.remote([x_test[batchsize*x:batchsize*(x+1)], y_test[batchsize*x:batchsize*(x+1)]], batchsize))
            loss += batch_loss
            acc += batch_acc
        print("Iter {}: \t loss is {} and accuracy of {}".format(i, (loss/len(x_test)), (acc/len(x_test))))

ray.shutdown()
