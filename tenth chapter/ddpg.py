import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import random
RENDER = False
C_UPDATE_STEPS = 1
A_UPDATE_STEPS = 1

class Experience_Buffer():
    def __init__(self,buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add_experience(self,experience):
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size]=[]
        self.buffer.extend(experience)
    def sample(self, samples_num):
        sample_data = np.reshape(np.array(random.sample(self.buffer, samples_num)),[samples_num, 4])
        train_s = np.array(sample_data[0,0])
        train_s_ = np.array(sample_data[0,3])
        train_a = sample_data[:, 1]
        train_r = sample_data[:, 2]
        for i in range(samples_num-1):
            train_s = np.vstack((train_s, np.array(sample_data[i+1,0])))
            train_s_ = np.vstack((train_s_, np.array(sample_data[i+1,3])))
        train_s = np.reshape(train_s,[samples_num,3])
        train_s_ = np.reshape(train_s_,[samples_num,3])
        train_r = np.reshape(train_r, [samples_num,1])
        train_a = np.reshape(train_a,[samples_num,1])
        return train_s, train_a, train_r, train_s_
#定义策略网络
class Policy_Net():
    def __init__(self, env, action_bound, lr = 0.0001, model_file=None):
        self.action_bound = action_bound
        self.gamma = 0.90
        self.tau = 0.01
        #  tf工程
        self.sess = tf.Session()
        self.learning_rate = lr
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        #输出动作空间的维数
        self.n_actions = 1
        #1. 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.obs_ = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #2.创建网络模型
        #2.1 创建策略网络，策略网络的命名空间为: 'actor'
        with tf.variable_scope('actor'):
            #可训练的策略网络,可训练的网络参数命名空间为: actor/eval:
            self.action = self.build_a_net(self.obs, scope='eval', trainable=True)
            #靶子策略网络，不可训练,网络参数命名空间为：actor/target:
            self.action_=self.build_a_net(self.obs_, scope='target',trainable=False)
        #2.2 创建行为值函数网络，行为值函数的命名空间为: 'critic'
        with tf.variable_scope('critic'):
            #可训练的行为值网络，可训练的网络参数命名空间为:critic/eval
            Q = self.build_c_net(self.obs, self.action, scope='eval', trainable=True)
            Q_ = self.build_c_net(self.obs_, self.action_, scope='target', trainable=False)
        #2.3 整理4套网络参数
        #2.3.1：可训练的策略网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        #2.3.2: 不可训练的策略网络参数
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        #2.3.3: 可训练的行为值网络参数
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
        #2.3.4: 不可训练的行为值网络参数
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')
        #2.4 定义新旧参数的替换操作
        self.update_olda_op = [olda.assign((1-self.tau)*olda+self.tau*p) for p,olda in zip(self.ae_params, self.at_params)]
        self.update_oldc_op = [oldc.assign((1-self.tau)*oldc+self.tau*p) for p,oldc in zip(self.ce_params, self.ct_params)]
        #3.构建损失函数
        #3.1 构建行为值函数的损失函数
        self.R = tf.placeholder(tf.float32, [None, 1])
        Q_target = self.R + self.gamma * Q_
        self.c_loss = tf.losses.mean_squared_error(labels=Q_target, predictions=Q)
        #3.2 构建策略损失函数，该函数为行为值函数
        self.a_loss=-tf.reduce_mean(Q)
        #4. 定义优化器
        #4.1 定义动作优化器,注意优化的变量在ca_params中
        self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, var_list=self.ae_params)
        #4.2 定义值函数优化器，注意优化的变量在ce_params中
        self.c_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.c_loss, var_list=self.ce_params)
        #5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    def build_c_net(self,obs, action, scope, trainable):
        with tf.variable_scope(scope):
            c_l1 = 50
            #与状态相对应的权值
            w1_obs = tf.get_variable('w1_obs',[self.n_features, c_l1], trainable=trainable)
            #与动作相对应的权值
            w1_action = tf.get_variable('w1_action',[self.n_actions, c_l1],trainable=trainable)
            b1 = tf.get_variable('b1',[1, c_l1], trainable=trainable)
            #第一层隐含层
            c_f1 = tf.nn.relu(tf.matmul(obs, w1_obs)+tf.matmul(action,w1_action)+b1)
            # 第二层， 行为值函数输出层
            c_out = tf.layers.dense(c_f1, units=1, trainable=trainable)
        return c_out
    def build_a_net(self, obs, scope, trainable):
        with tf.variable_scope(scope):
            # 行为值网络第一层隐含层
            a_f1 = tf.layers.dense(inputs=obs, units=100, activation=tf.nn.relu, trainable=trainable)
            # 第二层， 确定性策略
            a_out = 2 * tf.layers.dense(a_f1, units=self.n_actions, activation=tf.nn.tanh, trainable=trainable)
            return tf.clip_by_value(a_out, action_bound[0], action_bound[1])
    #根据策略网络选择动作
    def choose_action(self, state):
        action = self.sess.run(self.action, {self.obs:state})
        # print("greedy action",action)
        return action[0]
    #定义训练
    def train_step(self, train_s, train_a, train_r, train_s_):
        for _ in range(A_UPDATE_STEPS):
            self.sess.run(self.a_train_op, feed_dict={self.obs:train_s})
        for _ in range(C_UPDATE_STEPS):
            self.sess.run(self.c_train_op, feed_dict={self.obs:train_s, self.action:train_a, self.R:train_r, self.obs_:train_s_})
        # 更新旧的策略网络
        self.sess.run(self.update_oldc_op)
        self.sess.run(self.update_olda_op)
        # return a_loss, c_loss
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
def policy_train(env, brain, exp_buffer, training_num):
    reward_sum = 0
    average_reward_line = []
    training_time = []
    average_reward = 0
    batch = 32
    # for i in range(training_num):
    #     sample_states,sample_actions, sample_rs = sample.sample_steps(32)
    #     a_loss,c_loss = brain.train_step(sample_states, sample_actions,sample_rs)
    for i in range(training_num):
        total_reward = 0
        #初始化环境
        observation = env.reset()
        done = False
        while True:
            #探索权重衰减
            var = 3*np.exp(-i/100)
            state = np.reshape(observation, [1,brain.n_features])
            #根据神经网络选取动作
            action = brain.choose_action(state)
            #给动作添加随机项，以便进行探索
            action = np.clip(np.random.normal(action, var), -2, 2)
            obeservation_next, reward, done, info = env.step(action)
            # 存储一条经验
            experience = np.reshape(np.array([observation,action[0],reward/10,obeservation_next]),[1,4])
            exp_buffer.add_experience(experience)
            if len(exp_buffer.buffer)>batch:
                #采样数据，并进行训练
                train_s, train_a, train_r, train_s_ = exp_buffer.sample(batch)
                #学习一步
                brain.train_step(train_s, train_a, train_r, train_s_)
            #推进一步
            observation = obeservation_next
            total_reward += reward
            if done:
                break
        if i == 0:
            average_reward = total_reward
        else:
            average_reward = 0.95*average_reward + 0.05*total_reward
        print("第%d次学习后的平均回报为：%f"%(i,average_reward))
        average_reward_line.append(average_reward)
        training_time.append(i)
        if average_reward > -300:
            break
    brain.save_model('./current_best_ddpg_pendulum')
    plt.plot(training_time, average_reward_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()
def policy_test(env, policy,test_num):
    for i in range(test_num):
        reward_sum = 0
        observation = env.reset()
        print("第%d次测试，初始状态:%f,%f,%f" % (i, observation[0], observation[1], observation[2]))
        # 将一个episode的回报存储起来
        while True:
            env.render()
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, 3])
            action = policy.choose_action(state)
            observation_, reward, done, info = env.step(action)
            reward_sum += reward
            # print(reward)
            # reward_sum += (reward+8)/8
            if done:
                print("第%d次测试总回报%f" % (i,reward_sum))
                break
            time.sleep(0.01)
            observation = observation_
    # return reward_sum

if __name__=='__main__':
    #创建环境
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    #力矩边界
    action_bound = [-env.action_space.high, env.action_space.high]
    #实例化策略网络
    brain = Policy_Net(env,action_bound)
    # 下载当前最好得模型进行测试
    # brain = Policy_Net(env,action_bound,model_file='./current_best_ddpg_pendulum')
    #经验缓存
    exp_buffer = Experience_Buffer()
    training_num = 500
    #训练策略网络
    policy_train(env, brain, exp_buffer,training_num)
    #测试训练的网络
    reward_sum = policy_test(env, brain,100)







