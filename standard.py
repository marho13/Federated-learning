import gym
import os
import torch
import ray
from datetime import datetime

import wandb

from modelFiles.PPO import PPO
import numpy as np
from rayFiles.worker import DataWorker
from rayFiles.parameter import ParameterServer
from modelFiles.model import small_size, mid_size, large_size

class runner:
    def __init__(self, env_name, max_ep_len, has_continuous_action_space, k_epochs, eps_clip, gamma, lr_actor, lr_critic,
                 action_std, action_std_decay_rate, min_action_std, action_std_decay_freq, log_f_name, netsize, netstring):

        random_seed = 0  # set random seed if required (0 = no random seed)
        self.has_continuous_action_space = has_continuous_action_space
        env = gym.make(env_name)

        num_agents = 8
        num_runs = 10

        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            np.random.seed(random_seed)

        print("training environment name : " + env_name)

        state_dim = env.observation_space.shape[0]

        if self.has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n

        directory = "PPO_preTrained/"

        self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, self.has_continuous_action_space, action_std, netsize)
        self.env = gym.make(env_name)

        self.time_step = 0
        self.episode = 0
        self.run_num_pretrained = 0

        self.max_ep_len = max_ep_len
        self.update_timestep = self.max_ep_len * 4

        self.action_std = action_std
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.action_std_decay_freq = action_std_decay_freq

        self.save_model_freq = int(1e5)
        self.checkpoint_path = directory + "{}/PPO_{}_{}_{}.pth".format(env_name, env_name, random_seed, self.run_num_pretrained)
        self.start_time = datetime.now().replace(microsecond=0)

        self.log_f = open(log_f_name, "w+")
        self.log_f.write('episode,timestep,reward\n')
        self.k_epoch = k_epochs
        self.episode_rew = []
        self.print_freq = 50
        self.num_runs = num_runs

        self.runInitializations(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize, netstring)

    def initialiseRun(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize):
        self.worker = [DataWorker.remote(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                         self.has_continuous_action_space, action_std, env_name, netsize) for _ in range(
            num_agents)]  # DataWorker.remote(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, self.has_continuous_action_space, action_std, env_name)
        self.ps = ParameterServer(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip,
                                  self.has_continuous_action_space, action_std, num_agents, netsize)
        self.episode = 0
        self.time_step = 0


    def runInitializations(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize, netstring):
        for a in range(self.num_runs):
            self.wandy = wandb.init(project=env_name + "-{}".format(netstring), group="Baseline-sum",
                                    config={
                                        "batch_size": 16,
                                        "learning_rate_critic": lr_critic,
                                        "learning_rate_actor": lr_actor,
                                        "dataset": env_name,
                                        "model": "standard",
                                    })
            print("Run {}:".format(a))
            self.initialiseRun(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std, env_name, num_agents, netsize)
            self.workerMax(maxEp=200, maxTi=1000)
            self.wandy.finish()
        self.wandy.finish()

    def workerRun(self):
        current_weights = self.ps.get_weights()
        r = ray.get([w.gatherReplay.remote() for w in self.worker])#ray.get(self.worker.gatherReplay())#
        rew = [a.item() for a in r]
        loss = []
        for _ in range(self.k_epoch):
            output = ray.get([w.getLoss.remote(current_weights) for w in self.worker])#ray.get(self.worker.getLoss.remote(current_weights))#
            grad = []
            for a in range(len(output)):
                grad.append(output[a][0])
                loss.append(output[a][1].item())
            current_weights = self.ps.calcAvgGrad(*grad)
            # current_weights = self.ps.rewardweighted(grad, rew)#self.ps.appGrad(g)#
        [w.memClear.remote() for w in self.worker]#self.worker.memClear()#.remote()
        [w.policyOldUp.remote() for w in self.worker]#self.worker.policyOldUp()
        return (sum(rew)/len(rew)), (sum(loss)/len(loss))

    def workerMax(self, maxEp=1e11, maxTi=1e14):
        rew, loss = [], []
        while (self.episode < (maxEp+1)) and (self.time_step < (maxTi+1)):
            if self.episode > maxEp or self.time_step > maxTi:
                self.log_f.close()
                self.env.close()
                # print total training time
                print("============================================================================================")
                end_time = datetime.now().replace(microsecond=0)
                print("Started training at (GMT) : ", self.start_time)
                print("Finished training at (GMT) : ", end_time)
                print("Total training time  : ", end_time - self.start_time)
                print("============================================================================================")
            else:
                r, l = self.workerRun()
                self.wandy.log({"Reward":r}, step=self.episode)
                self.wandy.log({"Loss":l}, step=self.episode)
                loss.append(l)
                rew.append(r)
                self.episode += 1
                if self.episode%self.print_freq == 0:
                    avg = sum(rew[-self.print_freq:])/len(rew[-self.print_freq:])
                    # print(len(rew), rew[-100:], avg)
                    print("Episode {} gave reward {}".format(self.episode, avg))

env_name = "CartPole-v1"
max_ep_len = 1000
has_continuous_action_space = False

k_epochs = 10
eps_clip = 0.2
gamma = 0.99

lr_actor = 0.0003
lr_critic = 0.001

action_std = 0.6
action_std_decay_rate = 0.05
min_action_std = 0.1
action_std_decay_freq = int(2.5e5)

print(os.listdir())
log_f_name = "PPO_logs/{}/PPO_{}_log_1.csv".format(env_name, env_name)
print(log_f_name)

ray.init()
sizes = [small_size, mid_size, large_size]
size_string = ["small_size", "mid_size", "large_size"]
for i in range(len(sizes)):
    r = runner(env_name, max_ep_len, has_continuous_action_space, k_epochs, eps_clip, gamma, lr_actor, lr_critic,
               action_std, action_std_decay_rate, min_action_std, action_std_decay_freq, log_f_name, sizes[i], size_string[i])