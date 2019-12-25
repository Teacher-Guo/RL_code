import numpy as np
import tensorflow as tf
import random
import math
y= np.zeros((3,2))
print("y_",y[:,1:2])
x = np.array([[1,2,3]])
xx = x[0,:].tolist()
print(np.argmax(xx))
print(math.atan2(math.sin(4),math.cos(4)))
ob,ob2,obs=[],[],[]
for _ in range(10):
    ob.append([1,2])
    ob2.append([2,3])
ob = np.array(ob)
obs.append(ob)

print(obs[0])
# class Traj_info():
#     def __init__(self):
#         self.lamda = np.zeros((2,2,3))
# class Alg():
#     def __init__(self):
#         self.traj_info = Traj_info()
# alg = Alg()
# print(alg.traj_info.lamda)
# class Person(object):
#     def __init__(self, name, gender):
#         self.name = name
#         self.gender = gender
#     def __call__(self, friend):
#         print("my name is %s"%self.name)
#         print("may friend is %s"%friend)
# p = Person('Bob', 'male')
# p('Tim')
# x = np.zeros((0,1,2))
# print(x)
# input1 = tf.placeholder(tf.float32,[1,1])
# input2 = tf.placeholder(tf.float32,[1,1])
# c =tf.add(input1, input2)
# d = tf.add(input1, c)
# m =[[3.0]]
# n = tf.add(m,m)
# f = tf.add(n,input1)
# with tf.Session() as sess:
#     print(sess.run(f, feed_dict={input1:[[2.0]],n:[[4.0]]}))
# A = np.array([[0.1,10,100],[0.2, 20, 200]])
# B= np.mean(A,0)
# C = np.array([0,1,2])
# # print(B.shape)
# # print(B.shape)
# # B=np.array([[0,5,50]]
# print(A-C)
# # # print(np.mean(A))
# # # print(np.std(A,0))
# # B = (A-np.mean(A,0))
# # print(np.mean(A,0))
# print(A[:,0:2])
