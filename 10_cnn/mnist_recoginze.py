from keras.datasets import  mnist
import  numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

# print("X_train.shape:"+str(X_train.shape))
# print("Y_train.shape:"+str(Y_train.shape))
# print("X_test.shape:"+str(X_test.shape))
# print("Y_test.shape:"+str(Y_test.shape))
# print(Y_train[0])
# plt.imshow(X_train[0],cmap='gray')
# plt.show()

X_train=X_train.reshape(60000,784)/255 #归一化
X_test=X_test.reshape(10000,784)/255

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)
model=Sequential()
model.add(Dense(units=256,activation='relu',input_dim=784))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
#交叉熵代价函数多分类categorical_crossentropy
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.05),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=50,batch_size=10240)
loss,accuracy=model.evaluate(X_test,Y_test)
print("loss"+str(loss))
print("accuracy"+str(accuracy))
