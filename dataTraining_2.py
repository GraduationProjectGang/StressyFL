import json
import time
import pprint
import keras
from openpyxl import Workbook
from ast import literal_eval
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import time
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
from keras.models import model_from_json

filePath_data = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\trainingData.csv'
filePath_stress = 'C:\\Users\\Team6\\Documents\\GitHub\\DataManufacture\\stressData.csv'

trainingData_x = []
trainingData_y = []

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

trainingData_x = (trainingData_x - trainingData_x.min(axis=0)) / (trainingData_x.max(axis=0) - trainingData_x.min(axis=0))
trainingData_x = np.reshape(trainingData_x, (4014, 5, 5))

last_accuracy = 0.7

while True:
       x_train,x_val,y_train,y_val = train_test_split(trainingData_x, trainingData_y, test_size = 0.25)

       y_train = np_utils.to_categorical(y_train)
       y_val = np_utils.to_categorical(y_val)
       one_hot_vec_size = y_train.shape[1]

       # json_file = open("model.json", "r")
       # loaded_model_json = json_file.read()
       # json_file.close()
       # loaded_model = model_from_json(loaded_model_json)

       loaded_model = keras.models.load_model('best_model.h5')

       loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       hist = loaded_model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_val, y_val))

       loss_and_metrics = loaded_model.evaluate(x_val, y_val, batch_size=1)
       print('## evaluation loss and_metrics ##')
       print(loss_and_metrics)

       if last_accuracy <= loss_and_metrics[1]:
              last_accuracy = loss_and_metrics[1]
              print("good")
              # Save the entire model to a HDF5 file
              loaded_model.save('best_model.h5')
