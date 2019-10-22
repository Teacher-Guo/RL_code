import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env_td import *
from yuanyang_env_td import YuanYangEnv

class TD_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        #值函数的初始值
        self.qvalue=np.zeros((len(self.yuanyang.states),len(self.yuanyang.actions)))
    #定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax=qfun[state,:].argmax()
        return self.yuanyang.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]
    #找到动作所对应的序号
    def find_anum(self,a):
        for i in range(len(self.yuanyang.actions)):
            if a==self.yuanyang.actions[i]:
                return i

    def sarsa(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states),len(self.yuanyang.actions)))
        #第一个大循环，产生了多少次实验
        for iter in range(num_iter):
            #随机初始化状态
            epsilon = epsilon*0.99
            s_sample = []
            #初始状态，s0,
            # s = self.yuanyang.reset()
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num)<2:
                    print("sarsa 第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("sarsa 第一次实现最短路径需要的迭代次数为：", iter)
                break
            #随机选初始动作
            # a = self.yuanyang.actions[int(random.random()*len(self.yuanyang.actions))]
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0
            #第二个循环，一个实验，s0-s1-s2-s1-s2-s_terminate
            while False==t and count < 30:
                #与环境交互得到下一个状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                #判断一下 是否是终止状态
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作,这个地方体现on-policy
                    a1 = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(a1)
                    # qlearning的更新公式
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                    # 利用td方法更新动作值函数，alpha
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                # 转到下一个状态
                s = s_next
                #行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue
    def qlearning(self,num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        #大循环
        for iter in range(num_iter):
            #随机初始化状态
            # s = yuanyang.reset()
            s=0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num)<2:
                    print("qlearning 第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("qlearning 第一次实现最短路径需要的迭代次数为：", iter)
                break
            s_sample = []
            #随机选初始动作
            # a = self.actions[int(random.random()*len(self.actions))]
            a = self.epsilon_greedy_policy(self.qvalue,s,epsilon)
            t = False
            count = 0
            while False==t and count < 30:
                #与环境交互得到下一个状态
                s_next, r, t = yuanyang.transform(s, a)
                # print(s)
                # print(s_next)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作，a1用greedy_policy
                    a1 = self.greedy_policy(self.qvalue, s_next)
                    a1_num = self.find_anum(a1)
                    # qlearning的更新公式TD(0)
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                    # 利用td方法更新动作值函数
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                s = s_next
                #行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
                # print(r)
        return self.qvalue
    def greedy_test(self):
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
        if s == 9 and step_num<21:
            flag = 2
        return flag

if __name__=="__main__":
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    qvalue1 = brain.sarsa(num_iter =5000,alpha = 0.1, epsilon = 0.8)
    # qvalue2=brain.qlearning(num_iter=5000, alpha=0.1, epsilon=0.1)
    #打印学到的值函数
    yuanyang.action_value = qvalue1
    ##########################################
    # 测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    path = []
    # 将最优路径打印出来
    while flag:
        # 渲染路径点
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue1, s)
        # a = agent.bolzman_policy(qvalue,s,0.1)
        print('%d->%s\t' % (s, a), qvalue1[s, 0], qvalue1[s, 1], qvalue1[s, 2], qvalue1[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
