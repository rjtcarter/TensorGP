import numpy as np
import tensorflow as tf
from SBBGP import SBBEngine as sbb

trainX = np.loadtxt('data/gisette_train.data')
trainY = np.loadtxt('data/gisette_train.labels')
testX = np.loadtxt('data/gisette_valid.data')
testY = np.loadtxt('data/gisette_valid.labels')

train = np.reshape(trainX, (trainX.shape[0], -1))
train = np.hstack((train, np.expand_dims(trainY, 1)))

test = np.reshape(testX, (testX.shape[0], -1))
test = np.hstack((test, np.expand_dims(testY, 1)))

train = np.expand_dims(train, 2)
test = np.expand_dims(test, 2)

actions = np.unique(train[:,-1])
print("Data Loaded!")

train = tf.convert_to_tensor(train, dtype=tf.float32)
test = tf.convert_to_tensor(test, dtype=tf.float32)

print("Data converted to tensor!")

eng = sbb.SBBEngine(200, actions, trainX=train[:,:-1], testX=test[:,:-1], trainY=train[:,-1], testY=test[:,-1], outDirectory = 'gisette',  recordPerformance = True)

print("Initialized SBB!")

import time
start_time = time.time()

file = open('time.txt', 'w')

for i in range(5000):
    print('GENERATION: ' + str(i))
    eng.runGeneration()
    file.write(str(time.time() - start_time))