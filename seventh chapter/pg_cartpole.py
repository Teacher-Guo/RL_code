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
                action = self.policy_net.choose_action(state)
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
                    #归一化处理
                    discouted_sum_reward -= np.mean(discouted_sum_reward)
                    discouted_sum_reward/= np.std(discouted_sum_reward)
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
        return batch_obs, batch_actions,batch_rs
#定义策略网络
class Policy_Net():
    def __init__(self, env, model_file=None):
        self.learning_rate = 0.01
        #输入特征的维数
        self.n_features = env.observation_space.shape[0]
        #输出动作空间的维数
        self.n_actions = env.action_space.n
        #1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #1.2.第一层隐含层
        self.f1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                             bias_initializer=tf.constant_initializer(0.1))
        #1.3 第二隐含层
        self.all_act = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                       bias_initializer=tf.constant_initializer(0.1))
        #1.4 最后一层softmax层
        self.all_act_prob = tf.nn.softmax(self.all_act)
        #1.5 监督标签
        self.current_act = tf.placeholder(tf.int32, [None,])
        self.current_reward = tf.placeholder(tf.float32, [None,])
        #2. 构建损失函数
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.current_act)
        self.loss = tf.reduce_mean(neg_log_prob*self.current_reward)
        #3. 定义一个优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #4. tf工程
        self.sess = tf.Session()
        #5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    #定义贪婪策略
    def greedy_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:state})
        action = np.argmax(prob_weights,1)
        # print("greedy action",action)
        return action[0]
    #定义训练
    def train_step(self, state_batch, label_batch, reward_batch):
        loss, _ =self.sess.run([self.loss, self.train_op], feed_dict={self.obs:state_batch, self.current_act:label_batch, self.current_reward:reward_batch})
        return loss
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
    #依概率选择动作
    def choose_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:state})
        #按照给定的概率采样
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        # print("action",action)
        return action
def policy_train(env, brain, sample, training_num):
    reward_sum = 0.0
    reward_sum_line = []
    training_time = []
    for i in range(training_num):
        temp = 0
        training_time.append(i)
        # 采样10个episode
        train_obs, train_actions, train_rs = sample.sample_episodes(10)
        # 利用采样的数据进行梯度学习
        loss = brain.train_step(train_obs, train_actions, train_rs)
        # print("current loss is %f"%loss)
        if i == 0:
            reward_sum = policy_test(env, brain,False,1)
        else:
            reward_sum = 0.9 * reward_sum + 0.1 * policy_test(env, brain,False, 1)
        # print(policy_test(env, brain,False,1))
        reward_sum_line.append(reward_sum)
        print(reward_sum)
        print("training episodes is %d,trained reward_sum is %f" % (i, reward_sum))
        if reward_sum > 199:
            break
    brain.save_model('./current_bset_pg_cartpole')
    plt.plot(training_time, reward_sum_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()
def policy_test(env, policy, render, test_num):
    for i in range(test_num):
        observation = env.reset()
        reward_sum = 0
        # 将一个episode的回报存储起来
        while True:
            if render: env.render()
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, 4])
            action = policy.greedy_action(state)
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
    brain = Policy_Net(env)
    #实例化采样函数
    sampler = Sample(env,brain)
    #训练次数
    training_num = 150
    #训练策略网络
    policy_train(env, brain, sampler, training_num)
    #测试策略网络，随机生成10个初始状态进行测试
    reward_sum = policy_test(env, brain, True, 10)




