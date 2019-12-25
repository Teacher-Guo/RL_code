import tensorflow as tf
import numpy as np
import math
import gym
import matplotlib.pyplot as plt
RENDER = False
import random
#该程序将MPC嵌入到神经网络的训练，类似DAGGER思想
class Experience_Buffer():
    def __init__(self,buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add_experience(self,experience):
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size]=[]
        self.buffer.extend(experience)
    def sample(self, samples_num):
        sample_data = np.reshape(random.sample(self.buffer, samples_num),[samples_num,7])
        input = sample_data[:,0:4]
        label = sample_data[:,4:7]
        return input, label
#利用当前策略进行采样，产生数据
class Sample():
    def __init__(self,env, controller):
        self.env = env
        self.policy_net=controller
        self.gamma = 0.90
        self.n_features=3
        self.reward_sum = 0
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
        batch_reward = []
        for i in range(num_episodes):
            observation = self.env.reset()
            self.reward_sum = 0
            #将一个episode的回报存储起来
            reward_episode = []
            while True:
                # if RENDER:self.env.render()
                self.env.render()
                #根据策略网络产生一个动作
                state = np.reshape(observation,[1,3])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                reward_episode.append(reward)
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
                    self.reward_sum = np.sum(reward_episode)
                    print("本次总回报为%f"%self.reward_sum)
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.n_features])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.n_features])
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

        # 拟合动力学
        def fit_dynamic(self):
            flag = 0
            # 采样数据，产生20000个数据
            batch_obs_act, batch_delta, _ = self.sampler.sample_episodes(100)
            # 处理数据，正则化数据
            train_obs_act = (batch_obs_act - self.obs_action_mean) / (self.obs_action_std)
            train_delta = (batch_delta - self.delta_mean) / (self.delta_std)
            N = train_delta.shape[0]
            train_indicies = np.arange(N)
            loss_line = []
            num = 0
            ls = 0
            # 训练神经网络
            for i in range(self.iter):
                np.random.shuffle(train_indicies)
                for j in range(int(math.ceil(N / self.batch))):
                    start_idx = j * self.batch % N
                    idx = train_indicies[start_idx:start_idx + self.batch]
                    self.sess.run([self.train_op],
                                  feed_dict={self.obs_action: train_obs_act[idx, :], self.delta: train_delta[idx, :]})
                    loss = self.sess.run([self.loss], feed_dict={self.obs_action: train_obs_act[idx, :],
                                                                 self.delta: train_delta[idx, :]})
                    loss_line.append(loss)
                    if num == 0:
                        ls = loss[0]
                    else:
                        ls = 0.95 * ls + 0.05 * loss[0]
                    num += 1

                    if ls < 0.0001:
                        flag = 1
                        break
                if flag == 1:
                    break
                print("第%d次实验,误差为%f" % (i, ls))
            # 保存模型
            self.save_model('./current_best_dynamic_fit_pendulum')
            # 显示训练曲线
            number_line = np.arange(len(loss_line))
            plt.plot(number_line, loss_line)
            plt.show()
            # 拟合动力学
    def train_dynamic(self, mybuffer):
        flag = 0
        iter = 10
        batch=128
        # 处理数据，正则化数据
        for i in range(iter):
            batch_obs_act, batch_delta = mybuffer.sample(batch)
            train_obs_act = (batch_obs_act - self.obs_action_mean) / (self.obs_action_std)
            train_delta = (batch_delta - self.delta_mean) / (self.delta_std)
            self.sess.run([self.train_op],feed_dict={self.obs_action: train_obs_act, self.delta: train_delta})
        # # 保存模型
        # self.save_model('./current_best_dynamic_fit_pendulum')
        # # 显示训练曲线
        # number_line = np.arange(len(loss_line))
        # plt.plot(number_line, loss_line)
        # plt.show()
    def prediction(self,s_a, target_state=None):
        #正则化数据
        norm_s_a = (s_a-self.obs_action_mean)/self.obs_action_std
        #利用神经网络进行预测
        delta = self.sess.run(self.predict, feed_dict={self.obs_action:norm_s_a})
        predict_out = delta*self.delta_std + self.delta_mean +s_a[:,0:3]
        return predict_out
    def accurate_show(self,s_a, target_state):
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
class Mpc_Controller():
    def __init__(self, horizon=20, num_simulated_paths = 200):
        self.horizon =horizon
        self.num_simulated_path = num_simulated_paths
    def choose_action(self, state, dynn):
        self.dyn_model = dynn
        #参数state应该是个list, 如[s_1,s_2,s_3]
        state = state[0,:].tolist()
        #根据MPC选择动作
        ob,ob_as,costs = [],[],[]
        #当前观测
        for _ in range(self.num_simulated_path):
            ob.append(state)
        ob = np.array(ob)
        for _ in range(self.horizon):
            ac = []
            for _ in range(self.num_simulated_path):
                #产生随机动作
                ac.append([4*random.random()-2])
            # print(np.array(ac).shape)
            # print(np.array(ob).shape)
            #整理数据
            ob_a =np.hstack((np.array(ob),np.array(ac)))
            #保存数据，用于计算代价,ob_as的数据格式为[array,array]
            ob_as.append(ob_a)
            ob = self.dyn_model.prediction(ob_a)
            ob = ob.tolist()
        costs = self.compute_cost(ob_as)
        j = np.argmax(costs)
        return [ob_as[0][j,3]]
    #单摆的代价函数
    def compute_cost(self, ob_as):
        cost = np.zeros((self.num_simulated_path,1))
        for i in range(self.num_simulated_path):
            for j in range(self.horizon):
                cost[i,0]+= -(math.atan2(ob_as[j][i,1],ob_as[j][i,0])**2+.1*ob_as[j][i,2]**2+0.001*ob_as[j][i,3]**2)
        cost_sum = cost[:,0].tolist()
        return cost_sum


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
    #实例化动力学网络
    dynamic_net = Dynamic_Net(env, sampler)
    #MPC控制器
    mpc = Mpc_Controller()
    #实例化经验存储器
    mybuffer =  Experience_Buffer()
    average_reward = 0
    for i in range(100):
        #重置环境变量
        obs = env.reset()
        episode_reward = 0
        batch=128
        while True:
            env.render()
            current_state = np.reshape(obs, [1, 3])
            #MPC控制器
            action = mpc.choose_action(current_state,dynamic_net)
            obs_next, reward, done, info = env.step(action)
            #处理数据
            episode_reward+=reward
            delta = np.reshape(obs_next,[1,3])-current_state
            # print(current_state,action)
            obs_act = np.hstack((current_state, np.reshape(action, [1,1])))
            experience = np.hstack((obs_act, delta))
            mybuffer.add_experience(experience)
            if done:
                #训练神经网络
                dynamic_net.train_dynamic(mybuffer)
                break
            else:
                #前进一步
                obs = obs_next
        if i==0:
            average_reward = episode_reward
        else:
            print("episode_reward", episode_reward)
            average_reward = 0.95*average_reward+0.05*episode_reward
        print("第%d次实验，平均回报为%f"%(i, average_reward))
        if average_reward>-300:
            break










