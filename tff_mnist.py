import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import json
import time
import pprint
import keras
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime
import time
import csv
import ast
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

np.random.seed(0)

filePath_data = '\\home\\team6\\Documents\\GitHub\\StressyFL\\trainingData2.csv'
filePath_stress = '\\home\\team6\\Documents\\GitHub\\StressyFL\\stressData2.csv'

# 데이터 불러오기

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

trainingData_x = ((2 * (trainingData_x - trainingData_x.min(axis=0))) / (trainingData_x.max(axis=0) - trainingData_x.min(axis=0))) - 1
trainingData_x = np.reshape(trainingData_x, (4014, 5, 6))

# 클라이언트로 데이터 나누기

x_train,x_val,y_train,y_val = train_test_split(trainingData_x, trainingData_y, test_size = 0.3)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
one_hot_vec_size = y_train.shape[1]






NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER= 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
      x=tf.reshape(element['pixels'], [-1, 784]),
      y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))

print(sample_batch)

def make_federated_data(client_data, client_ids):
  return [
    preprocess(client_data.create_tf_dataset_for_client(x))
    for x in client_ids
  ]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

state = iterative_process.initialize()

state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))