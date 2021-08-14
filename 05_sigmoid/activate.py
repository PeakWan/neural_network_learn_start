# 加入激活函数sigmoid拟合有毒和无毒
import dataset
import matplotlib.pyplot as plt
import numpy as np

m = 100
xs, ys = dataset.get_beans(m)
# 初始化参数
w = 0.1
b = 0.1
# 随机梯度下降
for _ in range(5000):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 对w和b求偏导
        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        # 方差代价函数
        e = (y - a) ** 2
        # 链式求导
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x

        dedw = deda * dadz * dzdw
        dzdb = 1
        dedb = deda * dadz * dzdb
        alpha = 0.05
        # 反向传播
        w = w - alpha * dedw
        b = b - alpha * dedb
    if _ % 100 == 0:
        plt.clf()  # 清空窗口
        plt.title("Size-Toxicity Function", fontsize=12)
        plt.xlabel("Bean Size")
        plt.ylabel("Toxicity")
        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
        plt.plot(xs, a)
        plt.pause(0.01)
