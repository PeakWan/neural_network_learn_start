import numpy as np
import dataset
import plot_utils
m=100
X,Y=dataset.get_beans(m)

W=np.array([0.1,0.1])
B=np.array([0.1])
#前向传播
def forward_propgation(X):
    Z=X.dot(W.T)+B
    A=1/(1+np.exp(-Z))
    return A

for _ in range(500):
    for i in range(m):
        Xi=X[i]
        Yi=Y[i]
        A=forward_propgation(Xi)
        E=(Yi-A)**2
        dEdA=-2*(Yi-A)
        dAdZ=A*(1-A)
        dZdW=Xi
        dZdB=1
        #链式求导梯度
        dEdW=dEdA*dAdZ*dZdW
        dEdB = dEdA * dAdZ* dZdB
        #反向传播
        alpha=0.01
        W=W-alpha*dEdW
        B=B-alpha*dEdB
plot_utils.show_scatter_surface(X, Y, forward_propgation)
