import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m = 100
X, Y = dataset.get_beans4(m)
plot_utils.show_scatter(X, Y)

# Sequential 堆叠神经网络的载体
model = Sequential()
# Dense 全连接层 units:当前神经元数量 activation:激活函数 input_dim:特征维度
model.add(Dense(units=4, activation='sigmoid', input_dim=2))
model.add(Dense(units=1, activation='sigmoid'))
# loss代价函数:均方误差  optimizer优化器:SGD(随机梯度下降) metrics评估标准:准确率
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
# 训练5000回合 每次10 mini-batchsize
model.fit(X, Y, epochs=5000, batch_size=10)
pres = model.predict(X)
# plot_utils_08.show_scatter_curve(X,Y,pres)
plot_utils.show_scatter_surface(X, Y, model)
