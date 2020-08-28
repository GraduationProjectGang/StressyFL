import json
import time
import pprint
from openpyxl import Workbook
from ast import literal_eval
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import time
from rotate import getRotateVec
import csv
import tensorflow as tf
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.layers import Flatten
from keras import regularizers, optimizers
from keras.callbacks import LambdaCallback, ModelCheckpoint

filePath_data = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\trainingData2.csv'
filePath_stress = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\stressData2.csv'

trainingData_x = []
trainingData_y = []

# 정규화 [-1, 1]
# std_posture 추가
# test data 0.3 비율

with open(filePath_data, encoding= 'UTF-8') as file:
       data = csv.reader(file)

       for object in data:
              dummy_list = []
              for each in object:
                     each = ast.literal_eval(each)
                     map(float, each)
                     dummy_list.append(each)

              trainingData_x.append(dummy_list)

with open(filePath_stress, encoding= 'UTF-8') as file:
       data = csv.reader(file)
       for list in data:
              for stressCount in list:
                     trainingData_y.append(float(stressCount))

trainingData_x = np.array(trainingData_x)

trainingData_x = ((2 * (trainingData_x - trainingData_x.min(axis=0))) / (trainingData_x.max(axis=0) - trainingData_x.min(axis=0))) - 1
# trainingData_x = trainingData_x.tolist()
trainingData_x = np.reshape(trainingData_x, (4014, 5, 6))

# for i in range (0, len(trainingData_x)):
#        print(trainingData_x[i], " ", trainingData_y[i])

x_train,x_val,y_train,y_val = train_test_split(trainingData_x, trainingData_y, test_size = 0.3)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
one_hot_vec_size = y_train.shape[1]
print(y_train.shape[0], " ", y_train.shape[1], " ", y_train.shape, " ", one_hot_vec_size)
# print(y_train)

# 2. 모델 구성하기
# Dense란 ? 입력 하나에 출력 세 개 (첫번째가 학습해야할 weight 수, input dim이 입력) timestep 이 input length
# return sequences : 매 번 출력함
# stateful : 상태 유지 여부

print(x_train.shape)

model = Sequential()
model.add(LSTM(128, stateful=True, input_shape=x_train.shape, batch_input_shape=(1,5,6)))
model.add(Dropout(0.5))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 3. 모델 학습과정 설정하기
adam = optimizers.Adam(learning_rate=0.01, clipvalue=2.0)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기

# print(x_train)

# num_epochs = 2000
# for i in range(num_epochs):
#        print( 'epochs: {}'.format(i))
#        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, validation_data=(x_val, y_val), shuffle=False)
#        model.reset_states()

hist = model.fit(x_train, y_train, epochs=40, batch_size=1, validation_data=(x_val, y_val))
# hist = model.fit(x_train, y_train, epochs=10, batch_size=25, validation_data=(x_val, y_val), callbacks= [print_weights])

# 5. 학습과정 살펴보기

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_val, y_val, batch_size=10)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)


model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)

model.save('best_model_2_10.h5')