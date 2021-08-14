# 梯度下降
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)
# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)
w = 0.1
# y_pre = w * xs
# plt.plot(xs, y_pre)
# plt.show()

# 随机梯度下降
# for _ in range(2):
#     for i in range(100):
#         x = xs[i]
#         y = ys[i]
#         # a=x^2
#         # b=-2*x*y
#         # c=y^2
#         # 斜率k=2aw+b
#         k = 2 * (x ** 2) * w + (-2 * x * y)
#         # 学习率
#         alpha = 0.1
#         # 更新权重
#         w = w - alpha * k
#         plt.clf()  # 清空窗口
#         plt.scatter(xs, ys)
#         y_pre = w * xs
#         plt.xlim(0, 1)
#         plt.ylim(0, 1.2)
#         plt.plot(xs, y_pre)
#         plt.pause(0.01)

# 固定步长下降
# step = 0.01
# for _ in range(150):
#     # 代价函数:e=(y-w*x)^2
#     # a=x^2
#     # b=-2x*y
#     # 求解斜率:k=2aw+b
#     k = 2 * np.sum(xs ** 2) * w + np.sum(-2 * xs * ys)
#     if k - step > 0:
#         w = w - step
#     else:
#         w = w + step
#     y_pre = w * xs
#     plt.clf()
#     plt.xlim(0, 1)
#     plt.ylim(0, 1.2)
#     plt.plot(xs, y_pre)
#     plt.scatter(xs, ys)
#     plt.pause(0.01)

# 批量梯度下降
alpha = 0.1
for _ in range(100):
    k = 2 * np.sum(xs ** 2) * w + np.sum(-2 * xs * ys)
    k = k / 100
    w = w - alpha * k
    y_pre = w * xs
    plt.clf()
    plt.title("Size-Toxicity Function", fontsize=12)
    plt.xlabel("Bean Size")
    plt.ylabel("Toxicity")
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.scatter(xs, ys)
    plt.pause(0.01)
