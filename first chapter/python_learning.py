import numpy as np
from numpy import *
#python 类学习
####1.print语句学习################
print("hello reinforcement learning!")
####1.1打印字符串：%s##############
print('My name is %s'%('bao zi xian er'))
####1.2打印整数：%d####
print("I'm %d years old"%(20))
####1.3 打印浮点数：%f#####
print("I'm %f meters in height"%(1.80))
####1.4 打印浮点数，并保留两位小数：%.2f######
print("I'm %.2f meters in height"%(1.803))
###打印中英文混合############
print("老师how萌！")
####条件语句if...else...#######
score = 700
if score>700:
    print("上清华或北大")
else:
    print("复读")
########更细致的条件语句###########
score= 600
if score>700:
    print("上清华或北大")
elif score>=650:
    print("上其他双一流大学")
elif score>600 or score==600:
    print("上一本")
else:
    print("复读")
####3. 循环语句############
####3.1 for...in...依次将list或tuple中的每个元素迭代出来
a = [1,3,5,7,9,11]
for i in a:
    if i<10:
        print(i,"为10以内的奇数")
    else:
        print(i)
####更多例子########
b=["天","地","玄","黄"]
for i in b:
    print(i)
for i in range(10):
    print(i)
####3.2 while循环##########
# i=0
# while i<100:
#     print(i)
#     i+=1
####continue和break的应用
i=0
print("continue and break learning:")
while i<100:
    if i<50:
        i+=1
        continue
    print(i)
    i+=1
    if i>80:
        break
####4. 函数的定义###
print("函数定义学习")
def max(a,b):
    if a>b:
        return a
    else:
        return b
print(max(100,200))

#基类,普通类
print("基类和派生类学习")
class Car():
    def __init__(self, make,model,year):
        self.make=make
        self.model = model
        self.year = year
        self.odometer_reading = 0
    def get_descriptive_name(self):
        long_name = str(self.year)+' '+self.make+' '+self.model
        return long_name.title()
    def read_odometer(self):
        print("This car has"+str(self.odometer_reading)+"miles on it")
    def update_odometer(self,mileage):
        if mileage>=self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("you can't roll back an odometer!")
    def increment_odometer(self, miles):
        self.odometer_reading+=miles
#派生类
class ElectricCar(Car):
    def __init__(self,make,model,year):
        #基类初始化
        super().__init__(make, model,year)
####继承和派生########
my_tesla = ElectricCar('tesla','model s',2016)
print(my_tesla.get_descriptive_name())
#numpy库的基本操作
#####1.创建矩阵#####
A = np.array([[1,2,3],[2,4,6]])
print("创建的矩阵为：",A)
####创建向量
B=np.array([1,2,3])
print("创建的向量为：",B,B.shape)
#将向量变为矩阵
B=B[np.newaxis,:]
print("向量变为矩阵：",B,B.shape)
#创建全零矩阵
C=np.zeros((3,3))
print("全零矩阵为：",C)
#利用np.arange()创建整数数组
print("创建整数数组",np.arange(10))
#####矩阵的属性###
###矩阵维数##
print("The dimision of A is %d"%A.ndim)
###矩阵的形状#####
print("The shape of A is:",A.shape)
####矩阵的大小######
print("The size of A is:",A.size)
#####矩阵的函数操作######
print("转置操作：",A.transpose())
print("改变形状操作：",A.reshape((3,2)))
print("原矩阵：",A)
print("矩阵对应元素相乘：",A)
print("矩阵相乘：",np.dot(A,B.transpose()))
print("矩阵转置：",np.transpose(A))
print("访问矩阵的元素：",A[1,1],A[0,:],A[0,1:2],A[0,-1])
a1=A[0,:]
a2=A[1,:]
b1=A[:,0]
b2=A[:,1]
b3=A[:,2]
print("行合并",np.hstack((a1,a2)))
print("列合并",np.vstack((b1,b2,b3)))
print("常用的函数",np.sin(1.0),np.cos(1.0),np.exp(1.0))

