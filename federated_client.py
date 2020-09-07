import json
import time
import pprint
import keras
from ast import literal_eval
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import time
import csv
import tensorflow as tf
import ast
import os
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
from keras.models import model_from_json
import random
from tensorflow.keras.models import Model
from keras import backend as K
from sklearn.metrics import accuracy_score

# Failed to find the dnn implementation 오류 시

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# 0. Data 저장할 변수 선언

trainingData_x = []
trainingData_y = []

# 1. CSV 파일 읽어들이기

filePath_data = './trainingData_all.csv'
filePath_stress = './stressData_all.csv'

def import_data(data_path, stress_path):

    ret_x = []
    ret_y = []

    with open(data_path, encoding= 'UTF-8') as file:
        data = csv.reader(file)

        for object in data:
            dummy_list = []
            for each in object:
                each = ast.literal_eval(each)
                map(float, each)
                dummy_list.append(each)

            ret_x.append(dummy_list)

    with open(stress_path, encoding= 'UTF-8') as file:
        data = csv.reader(file)
        for list_ in data:
            for stressCount in list_:
                ret_y.append(float(stressCount))

    ret_x = np.array(ret_x)

    ret_x = ((2 * (ret_x - ret_x.min(axis=0))) / (ret_x.max(axis=0) - ret_x.min(axis=0))) - 1
    ret_x = np.reshape(ret_x, (4014, 5, 6))

    return ret_x, ret_y

# 2. Averaging Function 정의

# 2-1. weight_scaling_factor : 클라이언트 참여 비율 계산
def weight_scalling_factor(clients_trn_data, client_name):

    client_names = list(clients_trn_data.keys())
    
    # Batch Size 가져오기
    bs = list(clients_trn_data[client_name])[0][0].shape[0]

    # Client들의 총 Training Data 개수 계산 (모든 데이터 계산)
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    
    # Client가 보유한 총 Data 개수 가져오기
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs

    # Client 보유 데이터 개수 / 총 데이터 개수 (이 Client가 전체 데이터 중 어느 정도의 데이터를 제공하고 있는지에 대한 비율 return)
    return local_count/global_count

# 2-2. scale_model_weights : 모델 가중치를 조정하는 함수
# weight : 모델의 weight vector, scalar : weight_scaling_factor의 return value (해당 클라이언트의 참여비율)
def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

# 2-3. sum_scaled_weights : Scale된 weight들의 합 list를 return하는 함수
# scaled_weight_list : scale_model_weights 함수로부터 return된 가중치들을 계속 append한 list
def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()

    # 모든 클라이언트들의 Gradient (기울기) 를 가져온다.
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

# 3. Model 관련 함수, 클래스 정의

# 3-1. test_model : 모델 테스트하고 정보 출력
def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test, batch_size=1)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.5%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

# 3-2. Stressy_Model 클래스 정의 (모델 형태 설정, 일관성)
class Stressy_Model:
    @staticmethod
    def build():
        model = Sequential()
        model.add(LSTM(128, stateful=True, input_shape=x_train.shape, batch_input_shape=(1,5,6)))
        model.add(Dropout(0.5))
        model.add(Dense(one_hot_vec_size, activation='softmax'))
        return model

# 4. Client 관련 함수 생성

# 4-1. create_clients : Input, Output 데이터 가지고 Client를 생성
# return : 데이터 파편과 클라이언트 이름을 가진 dict
# data_list : Training data 리스트, label_list : 정답 레이블 리스트, num_client : 클라이언트 숫자
def create_clients(data_list, label_list, num_clients, initial='clients'):

    # Client 이름 생성 (List)
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    # Data를 섞음
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    # Client마다 Data 배정
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    # Client 숫자와 배정된 Data 같은지 확인
    assert(len(shards) == len(client_names))

    # Client와 Data를 엮은 Dict 객체 생성
    return {client_names[i] : shards[i] for i in range(len(client_names))}

# 4-2. batch_data : Client Data들을 가져와서 Tf.Dataset 객체 생성
# shard : 데이터, 라벨 묶음
# bs : 배치 사이즈
# return : tf.dataset 객체
def batch_data(data_shard, bs=1):
    #seperate shard into data and labels lists
    # 데이터 파편을 데이터와 레이블 리스트로 나눠
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

trainingData_x, trainingData_y = import_data(filePath_data, filePath_stress)

x_train,x_val,y_train,y_val = train_test_split(trainingData_x, trainingData_y, test_size = 0.1)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
one_hot_vec_size = y_train.shape[1]

clients = create_clients(x_train, y_train, num_clients=5)

clients_batched = dict()

for(client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

test_batched = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(len(y_val))

global_model = keras.models.load_model('78Model.h5')

lr = 0.01
comms_round = 1000


for comm_round in range(comms_round):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)
    
    #loop through each client and create new local model
    for client in client_names:
        stressy_local = Stressy_Model()
        local_model = stressy_local.build()
        local_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1)
        
        #scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)