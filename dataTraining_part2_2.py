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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

filePath_data = './trainingData_all.csv'
filePath_stress = './stressData_all.csv'

trainingData_x = []
trainingData_y = []

CLIENT_NUM = 32
client_idx = 0

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
    for list_ in data:
        for stressCount in list_:
            trainingData_y.append(float(stressCount))

trainingData_x = np.array(trainingData_x)

trainingData_x = ((2 * (trainingData_x - trainingData_x.min(axis=0))) / (trainingData_x.max(axis=0) - trainingData_x.min(axis=0))) - 1
trainingData_x = np.reshape(trainingData_x, (4014, 5, 6))

x_train,x_val,y_train,y_val = train_test_split(trainingData_x, trainingData_y, test_size = 0.1)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
one_hot_vec_size = y_train.shape[1]

# json_file = open("model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test, batch_size=1)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.5%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

class Stressy_Model:
    @staticmethod
    def build():
        model = Sequential()
        model.add(LSTM(128, stateful=True, input_shape=x_train.shape, batch_input_shape=(1,5,6)))
        model.add(Dropout(0.5))
        model.add(Dense(one_hot_vec_size, activation='softmax'))
        return model

# Create_Client 함수 정의

def create_clients(data_list, label_list, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}

clients = create_clients(x_train, y_train, num_clients=30)

# batch_data 함수 생성

def batch_data(data_shard, bs=1):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

clients_batched = dict()

for(client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

test_batched = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(len(y_val))

# Global Model 생성

# global_model = Sequential()
# global_model.add(LSTM(128, stateful=True, input_shape=x_train.shape, batch_input_shape=(1,5,6)))
# global_model.add(Dropout(0.5))
# global_model.add(Dense(one_hot_vec_size, activation='softmax'))

global_model = keras.models.load_model('78Model.h5')

# global_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = global_model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_val, y_val))

# loss_and_metrics = global_model.evaluate(x_val, y_val, batch_size=1)
# print('## evaluation loss and_metrics ##')
# print(loss_and_metrics)

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