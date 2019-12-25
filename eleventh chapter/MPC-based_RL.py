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
class Mpc_Sample():
    def __init__(self,env, mpc, dynn):
        self.env = env
        self.mpc = mpc
        self.dynamic_model = dynn
        self.gamma = 0.90
        self.n_features=3
        self.reward_sum = 0
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
                action = self.mpc.choose_action(state,self.dynamic_model)
                observation_, reward, done, info = self.env.step(action)
                reward_episode.append(reward)
                #存储当前观测
                batch_obs.append(np.reshape(observation,[1,3])[0,:])
                #存储后继观测
                batch_obs_next.append(np.reshape(observation_,[1,3])[0,:])
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
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        batch_obs_action= np.hstack((batch_obs,batch_actions))
        return batch_obs_action, batch_obs_next
class Dynamic_Net():
    def __init__(self,env, lr=0.0001, model_file=None):
        # 输入特征的维数
        self.n_features = env.observation_space.shape[0]
        self.learning_rate = lr
        self.obs_action_mean = np.array([0.0,0.0,0.0,0.0])
        self.obs_action_std = np.array([0.6303, 0.6708,3.5129, 1.1597])
        self.delta_mean = np.array([0.0, 0.0, 0.0])
        self.delta_std = np.array([0.1180, 0.1301, 0.5325])
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
    def train_dynamic(self, mybuffer):
        iter = 100
        batch=128
        # 处理数据，正则化数据
        for i in range(iter):
            batch_obs_act, batch_delta = mybuffer.sample(batch)
            train_obs_act = (batch_obs_act - self.obs_action_mean) / (self.obs_action_std)
            train_delta = (batch_delta - self.delta_mean) / (self.delta_std)
            self.sess.run([self.train_op],feed_dict={self.obs_action: train_obs_act, self.delta: train_delta})
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
def model_train(env, dynamic_net,mpc,mybuffer):
    #############开始训练##############################
    reward_line=[]
    average_line=[]
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
        reward_line.append(episode_reward)
        average_line.append(average_reward)
        print("第%d次实验，平均回报为%f"%(i, average_reward))
        if average_reward>-300:
            break
    x = np.arange(len(reward_line))
    plt.plot(x, reward_line)
    plt.plot(x,average_line,'--')
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()
    #########存储模型###########################
    dynamic_net.save_model('./current_best_trained_dynamic_pendulum')
def model_test(env, dynamic_net):
    #############测试模型的准确性###################
    mpc = Mpc_Controller()
    sampler_3 = Mpc_Sample(env, mpc, dynamic_net)
    batch_obs_act, target_state = sampler_3.sample_episodes(1)
    # print(sampler_3.reward_sum)
    predict_obs = dynamic_net.accurate_show(batch_obs_act, target_state)

if __name__=='__main__':
    # 创建仿真环境
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    # # 动作边界
    # action_bound = [-env.action_space.high, env.action_space.high]
    #实例化动力学网络
    dynamic_net = Dynamic_Net(env)
    #MPC控制器
    mpc = Mpc_Controller()
    #实例化经验存储器
    mybuffer =  Experience_Buffer()
    average_reward = 0
    #mpc训练
    model_train(env,dynamic_net,mpc,mybuffer)
    #测试学习到的模型
    model_test(env, dynamic_net)











