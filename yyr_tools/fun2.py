import matplotlib.pyplot as plt
import numpy as np
import math
# import torch
from scipy.optimize import minimize

x1=np.array( [0.938547,   0.9427869,  0.94238275, 0.98750585, 0.9722711,  0.9623701,
 1.0050269,  0.9185912,  1.0442636,  1.1389382 , 1.060165 ,  1.1331258,
 1.1513873,  1.1525736,  1.0089408,  1.056162 ,  1.1959628 , 1.1693922,
 0.98989177, 1.0226506 , 1.0109937 , 1.0500329 , 1.0601467 , 1.0818098,
 1.0383046,  1.0606034,  1.0937163])

y1=np.array( [0.8415974,  0.8625775,  0.86736494, 0.8646384,  0.8832723,  0.9006473,
 0.8583684,  0.94514763, 0.82418674, 0.51968503, 0.6108324,  0.54328907,
 0.5400955,  0.55348384, 0.7404691,  0.70677274, 0.6046301,  0.6420463,
 1.0445836,  1.0146844,  1.0333437,  1.0271394,  1.0360281,  1.0255831,
 1.0758268,  1.0554845 , 1.0287362])

mean_x = np.mean(x1)
mean_y = np.mean(y1)
# print(mean_x, mean_y)


# todo 绘制位面方程
dis = 0
b,c= np.mgrid[0:5:200j, -10:0:200j]
for i in range(len(x1)):
    dis1 = (np.abs(x1[i] + b * y1[i] + c) / np.sqrt(1 + b ** 2))
    dis = dis + dis1
ax = plt.axes(projection='3d')
ax.plot_surface(b, c, dis, rstride=1, cstride=1, cmap='rainbow', alpha =0.9)
plt.show()

# # todo 定义一个最优化的目标
# def disloss(p):
#     b, c = p
#     dis = 0
#     for i in range(len(x1)):
#         distance = np.abs(x1[i] + b * y1[i] + c) / np.sqrt(1 + b ** 2)
#         dis = dis + distance
#     return dis
#
# def disloss2(p):
#     b, c = p
#     dis = (np.abs(x1 + b * y1 + c) / np.sqrt(1 + b ** 2)).sum(0)
#     return dis
#
# # todo 调用优化算法求解的
# mymin1 = minimize(disloss, (1, mean_x + mean_y), method='BFGS')
# results1 = mymin1.x
# print(mymin1)
# print('***********************')
#
#
# mymin = minimize(disloss2, (1, mean_x + mean_y), method='Nelder-Mead')
# results = mymin.x
# print(results)


