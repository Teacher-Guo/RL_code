import numpy as np

phi_s_a = np.zeros((1,400))
theta = np.zeros((400,1))
qfun = np.array([0,1,2,2.5])
qfun[0]=10
print(np.dot(phi_s_a, theta)[0,0])
print(phi_s_a[0,1])
print(qfun.argmax())
print(np.transpose(phi_s_a))
print(np.reshape(theta, [100,4]))