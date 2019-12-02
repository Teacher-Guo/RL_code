import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
RENDER = False

#利用当前策略进行采样，产生数据
class Sample():
    def __init__(self,env, policy_net):
        self.env = env
        self.policy_net=policy_net
        self.gamma = 0.90
    def sample_episodes(self, num_episodes):
        #产生num_episodes条轨迹
        batch_obs_next = []
        batch_obs=[]
        batch_actions=[]
        batch_r =[]
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            # reward_episode = []
            while True:
                # if RENDER:self.env.render()
                #根据策略网络产生一个动作
                state = np.reshape(observation,[1,3])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                #存储当前观测
                batch_obs.append(np.reshape(observation,[1,3])[0,:])
                #存储后继观测
                batch_obs_next.append(np.reshape(observation_,[1,3])[0,:])
                # print('observation', np.reshape(observation,[1,3])[0,:])
                #存储当前动作
                batch_actions.append(action)
                #存储立即回报
                batch_r.append((reward+8)/8)
                # reward_episode.append((reward+8)/8)
                #一个episode结束
                if done:
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions,[len(batch_actions),])
        batch_rs = np.reshape(batch_r,[len(batch_r),1])
        return batch_obs, batch_obs_next,batch_actions,batch_rs
    def sample_step(self,observation):
        obs_next = []
        obs = []
        actions = []
        r = []
        state = np.reshape(observation, [1, 3])
        action = self.policy_net.choose_action(state)
        observation_, reward, done, info = self.env.step(action)
        # 存储当前观测
        obs.append(np.reshape(observation, [1, 3])[0, :])
        # 存储后继观测
        obs_next.append(np.reshape(observation_, [1, 3])[0, :])
        actions.append(action)
        # 存储立即回报
        r.append((reward+8)/8)
        # reshape 观测和回报
        obs = np.reshape(obs, [len(obs), self.policy_net.n_features])
        obs_next = np.reshape(obs, [len(obs), self.policy_net.n_features])
        actions = np.reshape(actions, [len(actions), ])
        r = np.reshape(r, [len(r), ])
        return obs, obs_next, actions, r, done

#定义策略网络
class Policy_Net():
    def __init__(self, env, action_bound, lr = 0.001, model_file=None):
        self.learning_rate = lr
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        #输出动作空间的维数
        self.n_actions = 1
        #1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #1.2.策略网络第一层隐含层
        self.a_f1 = tf.layers.dense(inputs=self.obs, units=200, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                             bias_initializer=tf.constant_initializer(0.1))
        #1.3 第二层，均值
        a_mu = tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        #1.3 第二层，标准差
        a_sigma = tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.softplus, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        #
        self.a_mu = 2*a_mu
        self.a_sigma =a_sigma
        # 定义带参数的正态分布
        self.normal_dist = tf.contrib.distributions.Normal(self.a_mu, self.a_sigma)
        #根据正态分布采样一个动作
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0],action_bound[1])
        #1.5 当前动作，输入为当前动作，delta,
        self.current_act = tf.placeholder(tf.float32, [None,])
        self.delta = tf.placeholder(tf.float32, [None,])
        #2. 构建损失函数
        log_prob = self.normal_dist.log_prob(self.current_act)
        self.a_loss = tf.reduce_mean(log_prob*self.delta)
        # self.loss += 0.01*self.normal_dist.entropy()
        #3. 定义一个动作优化器
        self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.a_loss)
        #4.定义critic网络
        self.c_f1 = tf.layers.dense(inputs=self.obs, units=100, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                             bias_initializer=tf.constant_initializer(0.1))
        self.v = tf.layers.dense(inputs=self.c_f1, units=1, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                             bias_initializer=tf.constant_initializer(0.1))
        #定义critic网络的损失函数,输入为td目标
        self.td_target = tf.placeholder(tf.float32, [None,1])
        self.c_loss = tf.losses.mean_squared_error(self.td_target,self.v)
        self.c_train_op = tf.train.AdamOptimizer(0.01).minimize(self.c_loss)
        #5. tf工程
        self.sess = tf.Session()
        #6. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #7.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    #依概率选择动作
    def choose_action(self, state):
        action = self.sess.run(self.action, {self.obs:state})
        # print("greedy action",action)
        return action[0]
    #定义训练
    def train_step(self, state, state_next, label, reward):
        #构建delta数据
        gamma = 0.90
        td_target = reward + gamma*self.sess.run(self.v, feed_dict={self.obs:state_next})
        delta = td_target - self.sess.run(self.v, feed_dict={self.obs:state})
        delta = np.reshape(delta,[200,1])[0]
        c_loss, _ = self.sess.run([self.c_loss, self.c_train_op],feed_dict={self.obs: state, self.td_target: td_target})
        a_loss, _ =self.sess.run([self.a_loss, self.a_train_op], feed_dict={self.obs:state, self.current_act:label, self.delta:delta})
        return a_loss, c_loss
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
def policy_train(env, brain, sample, training_num):
    reward_sum = 0
    reward_sum_line = []
    training_time = []
    for i in range(training_num):
        current_state, next_state,current_action, current_r = sample.sample_episodes(1)
        a_loss,c_loss = brain.train_step(current_state,next_state, current_action,current_r)

        # if i%50 == 0:
        #     RENDER = True
        # else:
        # RENDER=False

        # print(loss.shape)
        print("current experiments%d,current loss is %f"%(i, a_loss))
        # if i == 0:
        #     reward_sum = policy_test(env, brain,RENDER)
        # else:
        #     reward_sum = 0.9 * reward_sum + 0.1 * policy_test(env, brain,RENDER)
        # # print(policy_test(env, brain))
        # reward_sum_line.append(reward_sum)
        # print("training episodes is %d,trained reward_sum is %f" % (i, reward_sum))
        # if reward_sum > -1:
        #     break
    brain.save_model('./current_bset_pg_pendulum')
    plt.plot(training_time, reward_sum_line)
    plt.show()
def policy_test(env, policy,RENDER):
    observation = env.reset()
    reward_sum = 0
    # 将一个episode的回报存储起来
    while True:
        if RENDER:
            env.render()
        # 根据策略网络产生一个动作
        state = np.reshape(observation, [1, 3])
        action = policy.choose_action(state)
        observation_, reward, done, info = env.step(action)
        # print(reward)
        reward_sum += (reward+8)/8
        if done:
            # 处理回报函数
            break
        observation = observation_
    return reward_sum


#定义优化器
# class Optimizer()
#     def __init__(self):

if __name__=='__main__':
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    action_bound = [-env.action_space.high, env.action_space.high]
    brain = Policy_Net(env,action_bound)
    sampler = Sample(env, brain)
    training_num = 1500
    policy_train(env, brain, sampler, training_num)
    reward_sum = policy_test(env, brain,True)
    print("test reward_sum is %f"%reward_sum)