import numpy as np

def get_beans(counts):
	#生成随机数
	xs = np.random.rand(counts)
	#从小到大排序
	xs = np.sort(xs)
	ys = [1.2*x+np.random.rand()/10 for x in xs]
	return xs,ys

