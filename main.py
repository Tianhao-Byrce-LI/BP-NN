from keras.wrappers import scikit_learn
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ReduceLROnPlateau
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

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.datasets import mnist
from keras.utils import np_utils

import os
os.environ["PYTHONHASHSEED"] = '0'
import random as rn
import tensorflow as tf

np.random.seed(50)
rn.seed(12346)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# 转成DataFrame格式方便数据处理
x_train_pd = pd.read_excel("xxl.xlsx")
y_train_pd = pd.read_excel("yxl.xlsx")
x_valid_pd = pd.read_excel("xcs.xlsx")
y_valid_pd = pd.read_excel("ycs.xlsx")
# print(x_train_pd.head())
# print('-------------------')
# print(y_train_pd.head())



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



def create_model(neurons_1=10,neurons_2=30):
    model = Sequential()  # 初始化，很重要！
    model.add(Dense(units = neurons_2,   # 输出大小
                    activation='relu',  # 激励函数
                    input_shape=(x_train_pd.shape[1],),  # 输入大小, 也就是列的大小
                    # kernel_initializer = 'random_uniform' # 初始权重的设定方法(如何随机)
                    )
              )
    
    
    model.add(Dropout(0.2))  # 丢弃神经元链接概率
    
    model.add(Dense(units = neurons_2,
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
                  optimizer=keras.optimizers.sgd(lr=0.1),  # 优化器
                  
                 )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=50, mode='auto')#设定学习率
    
    history = model.fit(x_train, y_train,
              epochs=100, # 迭代次数
              batch_size=50,  # 每次用来梯度下降的批处理数据大小
              verbose=0,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
              validation_data = (x_valid, y_valid),  # 验证集
              #callbacks=[reduce_lr]
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
       #反归一化
   # print(y_new.shape)
    y_new=y_new.reshape(24,1)
    min_max_scaler.fit(y_valid_pd)
    y_new = min_max_scaler.inverse_transform(y_new)
    
    print("预测值","\n",y_new)
    mape=abs(y_new-y_valid_pd)*100/y_valid_pd
    print("mape",mape)
    b=mape.mean()
    print("平均mape",b)
    return model

create_model()
##下面调参
# model = scikit_learn.KerasRegressor(build_fn=create_model,verbose=0)

# batch_size = [30,50,100,300]
# epochs = [100,300,500]
# learning_rate = [0.2, 0.3,0.4,0.5,0.6]
# momentum = [0.1,0.2,0.3,0.4,0.5,0.65,0.8,0.9,0.95]
# neurons_1 = [1, 5, 10, 15, 20, 25, 30]
# neurons_2 = [1, 5, 10, 15, 20, 25, 30]

# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,verbose=0)
# grid_result = grid.fit(x_train, y_train)

# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, std, param))
