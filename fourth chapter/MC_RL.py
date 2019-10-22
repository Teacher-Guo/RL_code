import pygame
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from yuanyang_env_mc import YuanYangEnv
class MC_RL:
    def __init__(self, yuanyang):
        #行为值函数的初始化
        self.qvalue = np.zeros((len(yuanyang.states),len(yuanyang.actions)))*0.1
        #次数初始化
        #n[s,a]=1,2,3?? 求经验平均时，q(s,a)=G(s,a)/n(s,a)
        self.n = 0.001*np.ones((len(yuanyang.states),len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
    # 定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]
    #定义e-贪婪策略,蒙特卡罗方法，要评估的策略时e-greedy策略，产生数据的策略。
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        #概率部分
        if np.random.uniform() < 1- epsilon:
            #最优动作
            return self.actions[amax]
        else:
            return self.actions[int(random.random()*len(self.actions))]
    #找到动作所对应的序号
    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i
    def mc_learning_on_policy(self, num_iter, epsilon):
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions)))
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        #学习num_iter次
        for iter1 in range(num_iter):
            #采集状态样本
            s_sample = []
            #采集动作样本
            a_sample = []
            #采集回报样本
            r_sample = []
            # #随机初始化状态
            # s = self.yuanyang.reset()
            #固定初始状态
            s = 0
            done = False
            step_num = 0
            epsilon = epsilon*np.exp(-iter1/1000)
            #采集数据s0-a1-s1-a2-s2...terminate state
            #for i in range(5):
            while False == done and step_num < 30:
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                #与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                #往回走给予惩罚
                if s_next in s_sample:
                    r= -2
                #存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                step_num+=1
                #转移到下一个状态，继续试验，s0-s1-s2
                s = s_next
            #任务完成结束条件
            if s == 9:
                print("同策略第一次完成任务需要的次数：", iter1)
                break
            #从样本中计算累计回报,g(s_0) = r_0+gamma*r_1+gamma^2*r_2+gamma^3*r3+v(sT)
            a = self.epsilon_greedy_policy(self.qvalue, s,epsilon)
            g = self.qvalue[s,self.find_anum(a)]
            #计算该序列的第一状态的累计回报
            for i in range(len(s_sample)-1, -1, -1):
                g *= self.gamma
                g += r_sample[i]
            # print("episode number, trajectory", iter1, s_sample)
            # print("first state", s_sample[0], g)
            #g=G(s1,a),开始算其他状态处的累计回报
            for i in range(len(s_sample)):
                #计算状态-行为对（s,a)的次数，s,a1...s,a2
                self.n[s_sample[i], a_sample[i]] += 1.0
                #利用增量式方法更新值函数
                self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]]*(self.n[s_sample[i],a_sample[i]]-1)+g)/ self.n[s_sample[i], a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
                # print("s_sample,a",g)
            # print("number",self.n)
        return self.qvalue
    def mc_learning_ei(self, num_iter):
        # 学习num_iter次
        self.qvalue=np.zeros((len(yuanyang.states), len(yuanyang.actions)))
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        for iter1 in range(num_iter):
            # 采集状态样本
            s_sample = []
            # 采集动作样本
            a_sample = []
            # 采集回报样本
            r_sample = []
            # 随机初始化状态
            s = self.yuanyang.reset()
            a = self.actions[int(random.random()*len(self.actions))]
            done = False
            step_num = 0
            if self.mc_test() == 1:
                print("探索初始化第一次完成任务需要的次数：", iter1)
                break
            # 采集数据s0-a1-s1-a2-s2...terminate state
            # for i in range(5):
            while False == done and step_num < 30:
                # 与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 往回走给予惩罚
                if s_next in s_sample:
                    r = -2
                # 存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                step_num += 1
                # 转移到下一个状态，继续试验，s0-s1-s2
                s = s_next
                a = self.greedy_policy(self.qvalue, s)
            # 从样本中计算累计回报,g(s_0) = r_0+gamma*r_1+gamma^2*r_2+gamma^3*r3+v(sT)
            a = self.greedy_policy(self.qvalue, s)
            g = self.qvalue[s, self.find_anum(a)]
            # 计算该序列的第一状态的累计回报
            for i in range(len(s_sample) - 1, -1, -1):
                g *= self.gamma
                g += r_sample[i]
            # print("episode number, trajectory", iter1, s_sample)
            # print("first state", s_sample[0], g)
            # g=G(s1,a),开始算其他状态处的累计回报
            for i in range(len(s_sample)):
                # 计算状态-行为对（s,a)的次数，s,a1...s,a2
                self.n[s_sample[i], a_sample[i]] += 1.0
                # 利用增量式方法更新值函数
                self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]] * (
                            self.n[s_sample[i], a_sample[i]] - 1) + g) / self.n[s_sample[i], a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
                # print("s_sample,a",g)
            # print("number",self.n)
        return self.qvalue
    def mc_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 9:
            flag = 1
        return flag

if __name__=="__main__":
    yuanyang = YuanYangEnv()
    brain = MC_RL(yuanyang)
    # 探索初始化方法
    qvalue1 = brain.mc_learning_ei(num_iter=10000)
    #on-policy方法
    qvalue2=brain.mc_learning_on_policy(num_iter=10000, epsilon=0.2)
    print(qvalue2)
    #将行为值函数渲染出来
    yuanyang.action_value = qvalue2
    ##########################################
    #测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    path = []
    # 将最优路径打印出来
    while flag:
        #渲染路径点
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue2,s)
        # a = agent.bolzman_policy(qvalue,s,0.1)
        print('%d->%s\t' % (s, a),qvalue2[s,0],qvalue2[s,1],qvalue2[s,2],qvalue2[s,3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    #渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()






