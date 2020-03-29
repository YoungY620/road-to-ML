import torch 
import torchvision
from torchvision import datasets
from torch import nn
import gym
import numpy as np

GREEDY_RATE = 0.9               # for epsilon greedy. possibility of a = maxQ(s,a)
BUFFER_CAPABILITY = 5000
BATCH_SIZE = 100
LR = 0.01
N_EPISODES = 400
REPLACE_ITER = 100
REWARD_DISCOUNT = 0.9
NOISY_NET_RATE = 2
REWARD_EXAGGERATE = 2

env = gym.make('MsPacman-ram-v0')
# print(type(env))                      # <class 'gym.wrappers.time_limit.TimeLimit'>
env = env.unwrapped
# print(type(env))                      # <class 'gym.envs.classic_control.cartpole.CartPoleEnv'>
N_STATES = env.observation_space.shape[0]
# print(type(env.observation_space))    # <class 'gym.spaces.box.Box'>
# print(env.observation_space.shape)    # (128,)
N_ACTION = env.action_space.n
# print(type(env.action_space))         # <class 'gym.spaces.discrete.Discrete'>
# print(env.action_space)               # 9
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(N_STATES,64),
            nn.ReLU(),
            nn.Linear(64,32),
        )
        self.to_A = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,N_ACTION)
        )
        self.to_V = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.hidden(x)
        v = self.to_V(x)
        a = self.to_A(x)
        a = a-a.mean()+v[0]
        return a


class DQN(object):
    def __init__(self):
        super(DQN, self).__init__()
        self.refreshed_net, self.fixed_net = Net(), Net()
        self.optimizer = torch.optim.Adam(params=self.refreshed_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()

        self.replay_buffer = np.zeros((BUFFER_CAPABILITY,N_STATES*2+2))
        self.buffer_counter = 0
        self.learn_counter = 0
    def choose_action(self,s):
        s = torch.FloatTensor(s).unsqueeze(0)
        # print(s[:10])
        if np.random.rand()<GREEDY_RATE:
            action = self.fixed_net.forward(s).max(1)[1].numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:
            action = np.random.randint(0,N_ACTION)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    def store_buffer(self, s,a,r,s_):
        combine = np.hstack((s, [a, r], s_))
        index = self.buffer_counter%BUFFER_CAPABILITY
        self.replay_buffer[index,:] = combine
        self.buffer_counter += 1
    def learn(self):
        if self.learn_counter == REPLACE_ITER:
            self.fixed_net.load_state_dict(self.refreshed_net.state_dict())
        self.learn_counter += 1

        indexes = np.random.choice(BUFFER_CAPABILITY,BATCH_SIZE)
        train_data = self.replay_buffer[indexes,:]
        b_s = torch.FloatTensor(train_data[:,:N_STATES])                        # (batch_size, n_states)
        b_a = torch.LongTensor(train_data[:,N_STATES:N_STATES+1].astype(int))   # (batch_size, 1)
        b_r = torch.FloatTensor(train_data[:,N_STATES+1:N_STATES+2])            # (batch_size, 1)
        b_s_ = torch.FloatTensor(train_data[:,-N_STATES:])                      # (batch_size, n_states)

        # prepare data, double dqn
        q_eval = self.refreshed_net(b_s).gather(1,b_a)                      # (batch_size, 1)
        a_max = self.refreshed_net.forward(b_s_).max(1)[1].unsqueeze(1)     # (batch_size, 1)
        q_next = self.fixed_net(b_s_).detach().gather(1,a_max)              # 
        q_target = b_r + REWARD_DISCOUNT*q_next

        # train
        loss = self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    def add_noise(self):
        for param in self.fixed_net.parameters():
            param.data = param.data+(NOISY_NET_RATE*torch.randn(param.data.shape))


dqn = DQN()
print("collecting experience...")
for episode in range(N_EPISODES):
    s = env.reset()
    # print(type(s)," ",s.shape)
    q_r = 0
    done = False
    while not done: # one loop, one action
        env.render()
        action = dqn.choose_action(s)

        s_, r, done, _ = env.step(action)
        r = pow(r,REWARD_EXAGGERATE)
        dqn.store_buffer(s,action,r,s)

        q_r += pow(r,1/REWARD_EXAGGERATE)

        if dqn.buffer_counter>=BUFFER_CAPABILITY:
            loss = dqn.learn()
            dqn.add_noise()
            if done:
                print("episode: %5s"%episode,"|qr: %6.2f"%q_r,"|loss: %.4f"%loss)
        if not done:
            s = s_
