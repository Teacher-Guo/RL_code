import tensorflow as tf
import numpy as np
import math
import gym
import matplotlib.pyplot as plt
RENDER = False
import random
#利用当前策略进行采样，产生数据
class Sample():
    def __init__(self,env, policy_net):
        self.env = env
        self.policy_net=policy_net
        self.gamma = 0.90
    def sample_normalize(self, num_episodes):
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
                # action = self.policy_net.choose_action(state)
                action = [4*random.random()-2]
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
                    #处理数据
                    # print(self.delta)
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.n_features])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs_next), self.policy_net.n_features])
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        delta = batch_obs_next - batch_obs
        self.obs_mean, self.obs_std = self.normalize(batch_obs)
        self.delta_mean, self.delta_std = self.normalize(delta)
        self.action_mean, self.action_std = self.normalize(batch_actions)
        self.obs_action_mean = np.hstack((self.obs_mean, self.action_mean))
        self.obs_action_std = np.hstack((self.obs_std, self.action_std))
        return self.obs_action_mean, self.obs_action_std, self.delta_mean,self.delta_std
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
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        batch_rs = np.reshape(batch_r,[len(batch_r),1])
        batch_obs_action= np.hstack((batch_obs,batch_actions))
        return batch_obs_action, batch_delta,batch_obs_next
    def normalize(self, batch_data):
        mean = np.mean(batch_data,0)
        std = np.std(batch_data,0)
        return mean, std
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
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
class Dynamic_Net():
    def __init__(self,env, sampler, lr=0.0001, model_file=None):
        # 输入特征的维数
        self.n_features = env.observation_space.shape[0]
        self.learning_rate = lr
        self.sampler = sampler
        self.obs_action_mean = np.array([0.0,0.0,0.0,0.0])
        self.obs_action_std = np.array([0.6303, 0.6708,3.5129, 1.1597])
        self.delta_mean = np.array([0.0, 0.0, 0.0])
        self.delta_std = np.array([0.1180, 0.1301, 0.5325])
        # 得到数据的均值和协方差,产生100条轨迹
        self.sampler.sample_normalize(100)
        self.batch = 50
        self.iter = 2000
        # 输出动作空间的维数
        self.n_actions = 1
        # 1.1 输入层
        self.obs_action = tf.placeholder(tf.float32, shape=[None, self.n_features+self.n_actions])
        # 1.2.第一层隐含层100个神经元,激活函数为relu
        self.f1 = tf.layers.dense(inputs=self.obs_action, units=200, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1), \
                                  bias_initializer=tf.constant_initializer(0.1))
        # 1.3 第二层隐含层100个神经元，激活函数为relu
        self.f2 = tf.layers.dense(inputs=self.f1, units=100, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                  bias_initializer=tf.constant_initializer(0.1))
        # 1.4 输出层3个神经元，没有激活函数
        self.predict = tf.layers.dense(inputs=self.f2, units= self.n_features)
        # 2. 构建损失函数
        self.delta =  tf.placeholder(tf.float32,[None, self.n_features])
        self.loss = tf.reduce_mean(tf.square(self.predict-self.delta))
        # 3. 定义一个优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # 4. tf工程
        self.sess = tf.Session()
        # 5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        # 6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    #拟合动力学
    def fit_dynamic(self):
        flag = 0
        #采样数据，产生20000个数据
        batch_obs_act, batch_delta,_ = self.sampler.sample_episodes(100)
        #处理数据，正则化数据
        train_obs_act = (batch_obs_act-self.obs_action_mean)/(self.obs_action_std)
        train_delta = (batch_delta-self.delta_mean)/(self.delta_std)
        N = train_delta.shape[0]
        train_indicies = np.arange(N)
        loss_line=[]
        num = 0
        ls = 0
        #训练神经网络
        for i in range(self.iter):
            np.random.shuffle(train_indicies)
            for j in range(int(math.ceil(N/self.batch))):
                start_idx = j * self.batch%N
                idx = train_indicies[start_idx:start_idx+self.batch]
                self.sess.run([self.train_op], feed_dict={self.obs_action:train_obs_act[idx,:], self.delta:train_delta[idx,:]})
                loss = self.sess.run([self.loss],feed_dict={self.obs_action:train_obs_act[idx,:], self.delta:train_delta[idx,:]})
                loss_line.append(loss)
                if num == 0:
                    ls=loss[0]
                else:
                    ls = 0.95*ls+0.05*loss[0]
                num+=1

                if ls < 0.0001:
                    flag=1
                    break
            if flag == 1:
                break
            print("第%d次实验,误差为%f"%(i, ls))
        #保存模型
        self.save_model('./current_best_dynamic_fit_pendulum')
        #显示训练曲线
        number_line=np.arange(len(loss_line))
        plt.plot(number_line, loss_line)
        plt.show()
    def prediction(self,s_a, target_state):
        #正则化数据
        norm_s_a = (s_a-self.obs_action_mean)/self.obs_action_std
        #利用神经网络进行预测
        delta = self.sess.run(self.predict, feed_dict={self.obs_action:norm_s_a})
        predict_out = delta*self.delta_std + self.delta_mean +s_a[:,0:3]
        x = np.arange(len(predict_out))
        plt.figure(1)
        plt.plot(x, predict_out[:,0],)
        plt.plot(x, target_state[:,0],'--')
        # plt.figure(11)
        # plt.plot(x, s_a[:,0])
        # plt.plot(x,predict_out[:,0],'--')
        # plt.plot(x, target_state[:,0],'-.')
        plt.figure(2)
        plt.plot(x, predict_out[:, 1])
        plt.plot(x, target_state[:, 1],'--')
        plt.figure(3)
        plt.plot(x, predict_out[:, 2])
        plt.plot(x, target_state[:, 2],'--')
        plt.show()
        return predict_out
    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

if __name__=='__main__':
    # 创建仿真环境
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    # 动作边界
    action_bound = [-env.action_space.high, env.action_space.high]
    # 实例化策略网络
    brain = Policy_Net(env, action_bound)
    # 实例化采样
    sampler = Sample(env, brain)
    #随机采样200条轨迹
    # print(sampler.sample_normalize(200))
    # print(sampler.obs_action_mean, sampler.obs_action_std)
    # xs, xy= sampler.sample_episodes(1)
    # print(xs.shape, xy.shape)
    dynamic_net = Dynamic_Net(env, sampler,model_file='./current_best_dynamic_fit_pendulum')
    # dynamic_net = Dynamic_Net(env, sampler)
    # #拟合
    # dynamic_net.fit_dynamic()
    #测试初始策略拟合的动力学
    batch_obs_act, batch_delta, target_state = sampler.sample_episodes(1)
    predict_obs = dynamic_net.prediction(batch_obs_act,target_state)
    #测试ppo策略产生的数据对拟合模型的误差
    brain_2 = Policy_Net(env, action_bound,model_file='./current_best_ppo_pendulum')
    sampler_2 = Sample(env, brain_2)
    batch_obs_act, batch_delta, target_state = sampler_2.sample_episodes(10)
    predict_obs = dynamic_net.prediction(batch_obs_act, target_state)


    # print(predict_obs)





