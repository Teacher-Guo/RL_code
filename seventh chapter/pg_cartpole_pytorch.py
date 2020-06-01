import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
# @misc{pytorchrl,
#   author = {Kostrikov, Ilya},
#   title = {PyTorch Implementations of Reinforcement Learning Algorithms},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
# }
RENDER = False
#调用pytorch的分布函数
FixedCategorical = torch.distributions.Categorical
#重新定义原来的分布类的成员函数
old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self:old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs=lambda self, actions:log_prob_cat(self,actions.squeeze(-1)).view(actions.size(0),-1).sum(-1).unsqueeze(-1)
FixedCategorical.mode = lambda self:self.probs.argmax(dim=-1,keepdim=True)

#利用当前策略进行采样，产生数据
class Sample():
    def __init__(self,env, policy_net):
        self.env = env
        self.policy_net=policy_net
        self.gamma = 0.98
    def sample_episodes(self, num_episodes):
        #产生num_episodes条轨迹
        batch_obs=[]
        batch_actions=[]
        batch_rs =[]
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            reward_episode = []
            while True:
                if RENDER:self.env.render()
                #根据策略网络产生一个动作
                state = np.reshape(observation,[1,4])
                #将numpy数据转换成torch张量
                state = torch.from_numpy(state)
                state = torch.as_tensor(state, dtype=torch.float32)
                action = self.policy_net.act(state)
                action = action.numpy()[0, 0]
                observation_, reward, done, info = self.env.step(action)
                batch_obs.append(observation)
                batch_actions.append(action)
                reward_episode.append(reward)
                #一个episode结束
                if done:
                    #处理回报函数
                    reward_sum = 0
                    discouted_sum_reward = np.zeros_like(reward_episode)
                    for t in reversed(range(0, len(reward_episode))):
                        reward_sum = reward_sum*self.gamma + reward_episode[t]
                        discouted_sum_reward[t] = reward_sum
                    # # #归一化处理
                    discouted_sum_reward -= np.mean(discouted_sum_reward)
                    discouted_sum_reward/= np.std(discouted_sum_reward)
                    #discouted_sum_reward+=0.05
                    #将归一化的数据存储到批回报中
                    for t in range(len(reward_episode)):
                        batch_rs.append(discouted_sum_reward[t])
                    break
                #智能体往前推进一步
                observation = observation_
        #存储观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions,[len(batch_actions),])
        batch_rs = np.reshape(batch_rs,[len(batch_rs),])
        #将数据转换为torch数据
        batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        batch_actions = torch.as_tensor(batch_actions,dtype=torch.float32)
        batch_rs = torch.as_tensor(batch_rs,dtype=torch.float32)
        batch_rs = batch_rs.view(-1,1)
        return batch_obs, batch_actions, batch_rs
#构建基本的线性层
class MLPBase(nn.Module):
    def __init__(self,num_inputs, hidden_size=20):
        super(MLPBase,self).__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(num_inputs,hidden_size)
        nn.init.normal_(self.linear_1.weight,mean=0,std=0.1)
        nn.init.constant_(self.linear_1.bias,0.1)
        #self.actor = nn.Sequential(nn.Linear(num_inputs,hidden_size),nn.ReLU())
    def forward(self,inputs):
        x = inputs
        x = self.linear_1(x)
        hidden_actor = F.relu(x)
        return hidden_actor
    @property
    def output_size(self):
        return self.hidden_size
#构建分布层
class Categorical(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Categorical,self).__init__()
        self.linear_2 = nn.Linear(num_inputs,num_outputs)
        nn.init.normal_(self.linear_2.weight,mean=0,std=0.1)
        nn.init.constant_(self.linear_2.bias, 0.1)
    def forward(self,x):
        x = self.linear_2(x)
        return FixedCategorical(logits=x)
#定义策略及相关的操作
class Policy(nn.Module):
    def __init__(self, env, model_file=None):
        super(Policy,self).__init__()
        self.learning_rate = 0.01
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        print(self.n_features)
        #输出动作空间的维数
        self.n_actions = env.action_space.n
        #定义前向神经网络模型
        # 1.1 动作网络的特征提取
        self.base=MLPBase(self.n_features)
        #1.2 动作分布,创建一个类对象
        self.dist = Categorical(self.base.output_size,self.n_actions)
    def act(self, inputs, deterministic=False):
        #输出动作特征
        actor_features = self.base(inputs)
        #输出一个分布，类型是nn.module类型
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action =  dist.sample()
        action_log_probs = dist.log_probs(action)
        return action
    def evaluate_actions(self,inputs,action):
        #输出动作特征层
        actor_features = self.base(inputs)
        #输出分布层
        dist = self.dist(actor_features)
        # print("概率分布：", dist.probs)
        # print("action", action)
        #计算当前动作概率的对数
        action_log_probs = dist.log_probs(action)
        #计算当前动作的交叉熵
        # print("概率分布：",dist.probs)
        # print("action",action)
        # action=action.long()
        # loss1 = nn.CrossEntropyLoss()
        # cross_entropy = loss1(dist.probs, action)
        # print("交叉熵",cross_entropy)
        # return cross_entropy
        return action_log_probs
#策略训练方法
class Policy_Gradient():
    def __init__(self,actor, lr = 0.01):
        #1. 定义网络模型
        self.actor_net = actor
        #2. 定义优化器
        self.optimizer = optim.Adam(self.actor_net.parameters(),lr = lr)
    def update(self,obs_batch,actions_batch,reward_batch):
        #obs_batch, actions,reward_batch,= rollouts
        # print("action",actions_batch)
        #action_cross_entropy = self.actor_net.evaluate_actions(obs_batch,actions_batch)
        action_log_probs = -self.actor_net.evaluate_actions(obs_batch, actions_batch)
        #print("action_log_probs",action_log_probs)
        #reward_batch=reward_batch.view(-1,1)
        #print("reward",reward_batch)
        #获得损失函数的均值
        #loss =(action_cross_entropy*reward_batch).mean()

        loss_1 = action_log_probs*reward_batch
        #print("loss_1",loss_1.mean())
        loss = (action_log_probs * reward_batch).mean()
        #print("loss",action_log_probs*reward_batch)
        #print("loss",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#策略训练
def policy_train(env, alg, training_num):
    reward_sum = 0.0
    reward_sum_line = []
    training_time = []
    for i in range(training_num):
        sampler = Sample(env, alg.actor_net)
        temp = 0
        training_time.append(i)
        # 采样10个episode
        train_obs, train_actions, train_rs = sampler.sample_episodes(1)
        # 利用采样的数据进行梯度学习
        #print("train_rs",train_rs)
        alg.update(train_obs,train_actions,train_rs)
        # print("current loss is %f"%loss)
        if i == 0:
            reward_sum = policy_test(env, alg.actor_net,False,1)
        else:
            reward_sum = 0.9 * reward_sum + 0.1 * policy_test(env, alg.actor_net,False, 1)
        # print(policy_test(env, brain,False,1))
        reward_sum_line.append(reward_sum)
        print(reward_sum)
        print("training episodes is %d,trained reward_sum is %f" % (i, reward_sum))
        if reward_sum > 199:
            break
    #brain.save_model('./current_bset_pg_cartpole')
    plt.plot(training_time, reward_sum_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()

def policy_test(env, actor, render, test_num):
    for i in range(test_num):
        observation = env.reset()
        reward_sum = 0
        # 将一个episode的回报存储起来
        while True:
            if render: env.render()
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, 4])
            state=torch.as_tensor(state,dtype=torch.float32)
            action = actor.act(state,deterministic=True)
            action = action.numpy()[0, 0]
            observation_, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
            observation = observation_
    return reward_sum

if __name__=='__main__':
    #声明环境名称
    env_name = 'CartPole-v0'
    #调用gym环境
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    #下载当前最好的模型
    # brain = Policy_Net(env,'./current_bset_pg_cartpole')
    #实例化策略网络
    actor_1 = Policy(env)

    # observation = env.reset()
    # done = False
    # env.render()
    # while(done==False):
    #     env.render()
    #     state = np.reshape(observation,[1,4])
    #     state = torch.as_tensor(state, dtype=torch.float32)
    #     action = actor_net.act(state)
    #     action = action.numpy()[0,0]
    #     print("action",action)
    #     observation_, reward, done, info = env.step(action)
    # 将一个episode的回报存储起来
    #实例化采样函数

    pg = Policy_Gradient(actor_1,lr=0.01)
    #训练次数
    training_num = 15000
    #训练策略网络
    policy_train(env, alg=pg, training_num=training_num)
    #测试策略网络，随机生成10个初始状态进行测试
    reward_sum = policy_test(env, actor_1, True, 10)




