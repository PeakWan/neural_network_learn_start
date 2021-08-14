# 根据代价函数最低点w拟合样本
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)
# 产生连续的w值 [0,0.1,0.2,...,2.9]
ws = np.arange(0, 3, 0.1)
# 根据不同的w值代价函数值集合
es = []
for w in ws:
    # 预测函数
    y_pre = w * xs
    # 方差代价函数
    e = (1 / 100) * np.sum((ys - y_pre) ** 2)
    # 添加对应w点的方差代价函数值
    es.append(e)

# 配置w和方差代价函数的图形 【右侧曲线即为w与代价函数e的图形】
# plt.title("cost function", fontsize=12)
# plt.xlabel("w")
# plt.ylabel("e")
plt.plot(ws, es)
#plt.show()

# 利用推导的代价函数最低点公式直接求出最低点w的值
w_min = np.sum(xs * ys) / np.sum(xs * xs)
print("e最小点w:" + str(w_min)) # 【e最小点w:1.288787116859187  也就是拟合的直线斜率为emin对应的w值时拟合的最好】

# 根据最低点的代价函数w值拟合豆豆大小与毒性
y_pre = w_min * xs
# 绘制豆豆大小与毒性的图形
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)
plt.plot(xs, y_pre)
plt.show()
