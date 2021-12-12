import numpy as np
import tensorflow as tf
from SBBGP import SBBEngine as sbb


train = np.loadtxt('data/shuttletrain.txt')
test = np.loadtxt('data/shuttletest.txt')

train = np.expand_dims(train, 2)
test = np.expand_dims(test, 2)

actions = np.unique(train[:,-1])
print("Data Loaded!")

train = tf.convert_to_tensor(train, dtype=tf.float32)
test = tf.convert_to_tensor(test, dtype=tf.float32)

print("Data converted to tensor!")

eng = sbb.SBBEngine(200, actions, trainX=train[:,:-1], testX=test[:,:-1], trainY=train[:,-1], testY=test[:,-1], outDirectory = 'derp',  recordPerformance = True)

print("Initialized SBB!")

import time
start_time = time.time()

file = open('time.txt', 'w')

for i in range(5000):
    print('GENERATION: ' + str(i))
    eng.runGeneration()
    file.write(str(time.time() - start_time))