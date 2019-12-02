import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
RENDER = False
C_UPDATE_STEPS = 10
A_UPDATE_STEPS = 10
#利用当前策略进行采样，产生数据
class Sample():
    def __init__(self,env, policy_net):
        self.env = env
        self.gamma = 0.90
        self.brain = policy_net
    def sample_episodes(self, num_episodes):
        #产生num_episodes条轨迹
        batch_obs=[]
        batch_actions=[]
        batch_rs =[]
        #一次episode的水平
        batch = 200
        mini_batch = 32
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            reward_episode = []
            j = 0
            k = 0
            minibatch_obs = []
            minibatch_actions = []
            minibatch_rs = []
            while j < batch:
                #采集数据
                flag =1
                state = np.reshape(observation,[1,3])
                action = self.brain.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                #存储当前观测
                minibatch_obs.append(np.reshape(observation,[1,3])[0,:])
                #存储当前动作
                minibatch_actions.append(action)
                #存储立即回报
                minibatch_rs.append((reward+8)/8)
                k = k+1
                j = j+1
                if k==mini_batch or j==batch:
                    # 处理回报
                    reward_sum = self.brain.get_v(np.reshape(observation_, [1, 3]))[0, 0]
                    discouted_sum_reward = np.zeros_like(minibatch_rs)
                    for t in reversed(range(0, len(minibatch_rs))):
                        reward_sum = reward_sum * self.gamma + minibatch_rs[t]
                        discouted_sum_reward[t] = reward_sum
                    # 将mini批的数据存储到批回报中
                    for t in range(len(minibatch_rs)):
                        batch_rs.append(discouted_sum_reward[t])
                        batch_obs.append(minibatch_obs[t])
                        batch_actions.append(minibatch_actions[t])
                    k=0
                    minibatch_obs = []
                    minibatch_actions = []
                    minibatch_rs = []
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.brain.n_features])
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        batch_rs = np.reshape(batch_rs,[len(batch_rs),1])
        # print("batch_rs", batch_rs)
        return batch_obs,batch_actions,batch_rs

#定义策略网络
class Policy_Net():
    def __init__(self, env, action_bound, lr = 0.0001, model_file=None):
        tf.reset_default_graph()
        self.learning_rate = lr
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        #输出动作空间的维数
        self.n_actions = 1
        #1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.pi, self.pi_params = self.build_a_net('pi', trainable=True)
        self.oldpi, self.oldpi_params = self.build_a_net('oldpi', trainable=False)
        print("action_bound",action_bound[0],action_bound[1])
        self.action = tf.clip_by_value(tf.squeeze(self.pi.sample(1),axis=0), action_bound[0], action_bound[1])
        #定义新旧参数的替换操作
        self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(self.pi_params, self.oldpi_params)]
        #1.5 当前动作，输入为当前动作，delta,
        self.current_act = tf.placeholder(tf.float32, [None,1])
        #优势函数
        self.adv = tf.placeholder(tf.float32, [None,1])
        #2. 构建损失函数
        ratio = self.pi.prob(self.current_act)/self.oldpi.prob(self.current_act)
        #替代函数
        surr = ratio*self.adv
        self.a_loss = -tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio, 1.0-0.2, 1.0+0.2)*self.adv))
        # self.loss += 0.01*self.normal_dist.entropy()
        #3. 定义一个动作优化器
        self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss)
        #4.定义critic网络
        self.c_f1 = tf.layers.dense(inputs=self.obs, units=100, activation=tf.nn.relu)
        self.v = tf.layers.dense(inputs=self.c_f1, units=1)
        #定义critic网络的损失函数,输入为td目标
        self.td_target = tf.placeholder(tf.float32, [None,1])
        self.c_loss = tf.reduce_mean(tf.square(self.td_target-self.v))
        self.c_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.c_loss)
        # 5. tf工程
        self.sess = tf.Session()
        #6. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #7.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    def build_a_net(self, name, trainable):
        with tf.variable_scope(name):
            # 1.2.策略网络第一层隐含层
            self.a_f1 = tf.layers.dense(inputs=self.obs, units=100, activation=tf.nn.relu,trainable=trainable)
            # 1.3 第二层，均值
            a_mu = 2*tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.tanh,trainable=trainable)
            # 1.3 第二层，标准差
            a_sigma = tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.softplus,trainable=trainable)

            # a_mu = 2 * a_mu
            a_sigma = a_sigma
            # 定义带参数的正态分布
            normal_dist = tf.contrib.distributions.Normal(a_mu, a_sigma)
            # 根据正态分布采样一个动作
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return normal_dist, params
    def get_v(self, state):
        v = self.sess.run(self.v, {self.obs:state})
        return v
    #依概率选择动作
    def choose_action(self, state):
        action = self.sess.run(self.action, {self.obs:state})
        # print("greedy action",action)
        return action[0]
    #定义训练
    def train_step(self, state, label, reward):
        #更新旧的策略网络
        self.sess.run(self.update_oldpi_op)
        td_target = reward
        # print("reward",reward)
        delta = td_target - self.sess.run(self.v, feed_dict={self.obs:state})
        # print("delta",delta.shape)
        delta = np.reshape(delta,[len(delta),1])
        for _ in range(A_UPDATE_STEPS):
            self.sess.run([self.a_loss, self.a_train_op], feed_dict={self.obs:state, self.current_act:label, self.adv:delta})
        for _ in range(C_UPDATE_STEPS):
            self.sess.run([self.c_loss, self.c_train_op], feed_dict={self.obs: state, self.td_target: td_target})
        # return a_loss, c_loss
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
def policy_train(env, brain, sample, training_num):
    reward_sum = 0
    average_reward_line = []
    training_time = []
    average_reward = 0
    current_total_reward = 0
    for i in range(training_num):
        #采样
        current_state,current_action, current_r = sample.sample_episodes(1)
        #训练
        brain.train_step(current_state, current_action,current_r)
        current_total_reward = policy_test(env, brain,False,1)
        if i == 0:
            average_reward = current_total_reward
        else:
            average_reward = 0.95*average_reward + 0.05*current_total_reward
        average_reward_line.append(average_reward)
        training_time.append(i)
        if average_reward > -300:
            break
        print("current experiments%d,current average reward is %f"%(i, average_reward))
    brain.save_model('./current_best_ppo_pendulum')
    plt.plot(training_time, average_reward_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()
def policy_test(env, brain,RENDER, test_number):
    for i in range(test_number):
        observation = env.reset()
        if RENDER:
            print("第%d次测试，初始状态:%f,%f,%f" % (i, observation[0], observation[1], observation[2]))
        reward_sum = 0
        # 将一个episode的回报存储起来
        while True:
            if RENDER:
                env.render()
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, 3])
            action = brain.choose_action(state)
            observation_, reward, done, info = env.step(action)
            reward_sum+=reward
            # reward_sum += (reward+8)/8
            if done:
                if RENDER:
                    print("第%d次测试总回报%f" % (i, reward_sum))
                break
            observation = observation_
    return reward_sum

if __name__=='__main__':
    #创建仿真环境
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    #力矩界限
    action_bound = [-env.action_space.high, env.action_space.high]
    #实例化策略网络
    brain = Policy_Net(env,action_bound)
    #下载当前最好得模型进行测试
    # brain = Policy_Net(env,action_bound,model_file='./current_best_ppo_pendulum')
    #实例化采样
    sample = Sample(env, brain)
    #最大训练次数
    training_num = 5000
    #利用ppo算法训练神经网络
    policy_train(env, brain, sample, training_num)
    #对训练好的神经网络进行测试
    reward_sum = policy_test(env, brain,True,50)
