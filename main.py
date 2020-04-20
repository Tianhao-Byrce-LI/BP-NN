# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:56:14 2020

@author: aa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:07:56 2020

@author: aa
"""
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



# 转成DataFrame格式方便数据处理
x_train_pd = pd.read_excel("C:/Users/aa/Desktop/xxl.xlsx")
y_train_pd = pd.read_excel("C:/Users/aa/Desktop/yxl.xlsx")
x_valid_pd = pd.read_excel("C:/Users/aa/Desktop/xcs.xlsx")
y_valid_pd = pd.read_excel("C:/Users/aa/Desktop/ycs.xlsx")
#print(x_train_pd.head())
#print('-------------------')
#print(y_train_pd.head())

# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)


model = Sequential()  # 初始化，很重要！
model.add(Dense(units = 10,   # 输出大小
                activation='relu',  # 激励函数
                input_shape=(x_train_pd.shape[1],),  # 输入大小, 也就是列的大小
               # kernel_initializer = 'random_uniform' # 初始权重的设定方法(如何随机)
               )
         )

#model.add(Dropout(0.2))  # 丢弃神经元链接概率

model.add(Dense(units = 15,
#                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
#                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu', # 激励函数
                # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
              # kernel_initializer = 'random_uniform' # 初始权重的设定方法(如何随机)
                )
         )

model.add(Dense(units = 1,   
                activation='linear' , # 线性激励函数 回归一般在输出层用这个激励函数  
               #kernel_initializer = 'random_uniform' # 初始权重的设定方法(如何随机)
                )
         )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
              optimizer=keras.optimizers.sgd(lr=0.01),  # 优化器
             )

history = model.fit(x_train, y_train,
          epochs=2000,  # 迭代次数
          batch_size=100,  # 每次用来梯度下降的批处理数据大小
          verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_valid, y_valid)  # 验证集
        )


# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 预测
y_new = model.predict(x_valid)
# 反归一化

min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)

print("预测值","\n",y_new)

print("mape",abs(y_new-y_valid_pd)*100/y_valid_pd)
