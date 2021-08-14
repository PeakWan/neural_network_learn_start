import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten

x_train, y_train, x_test, y_test = shopping_data.load_data()
print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)
print(x_train[0])
print(y_train[0])

# vocalen 这个词典的词汇数量  word_index 训练集和测试集全部语料的词典
vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)
print(word_index)  # {'ok':1,'wifi':2.....,'灯':9459}
print('词典总词数:', vocalen)  # 词典总词数:9512

x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)

maxlen = 25
# 把序列按照maxlen进行对齐 长度不足25的句子就会被0补齐为25
x_train_index = sequence.pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen=maxlen)

model = Sequential()
# 嵌入层 trainable=True 是否让这一层在训练时更新参数 input_dim:输入维度 output_dim:输出维度 input_length:句子序列长度
model.add(Embedding(trainable=True, input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# loss:交叉熵代价函数  adam:是一种使用动量的自适应优化器
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_index, y_train, batch_size=512, epochs=100)
score, acc = model.evaluate(x_test_index, y_test)
print("Test score:", score)
print("Test accuracy:", acc)
