'''
罗森布拉特感知器模型
根据豆子大小预算毒性
'''
import dataset
from matplotlib import pyplot as plt

xs, ys = dataset.get_beans(100)
# 画出豆豆大小与毒性的散点图
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)
# 初始化权重w
w = 0.5
# 参数自适应调整
epoch = 100
for _ in range(epoch):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 预测函数
        y_pre = w * x
        # 代价函数
        e = y - y_pre
        # 学习率
        alpha = 0.05
        # 更新参数
        w = w + alpha * e * x
y_pre = w * xs
# 拟合豆豆线性图形
plt.plot(xs, y_pre)
plt.show()
