import numpy as np
import tensorflow as tf
from SBBGP import SBBEngine as sbb
import os

# Setting random seeds.
np.random.RandomState(42)
tf.random.set_seed(42)

train = np.loadtxt('data/shuttletrain.txt')
test = np.loadtxt('data/shuttletest.txt')

train = np.expand_dims(train, 2)
test = np.expand_dims(test, 2)

actions = np.unique(train[:,-1])
print("Data Loaded!")

train = tf.convert_to_tensor(train, dtype=tf.float32)
test = tf.convert_to_tensor(test, dtype=tf.float32)

print("Data converted to tensor!")


print("Initialized SBB!")

import time
start_time = time.time()


# With GPU
for i in range(30):

    start_time = time.time()

    try:
        os.makedirs('shuttleGPU/' + str(i))
    except OSError as error:
        print(error)

    eng = sbb.SBBEngine(200, actions, trainX=train[:,:-1], testX=test[:,:-1], trainY=train[:,-1], testY=test[:,-1], outDirectory = 'shuttleGPU/' + str(i),  recordPerformance = True, seed=42 + i, pRemoval=0.5, pAddition=0.95, numElites = 20)
    eng.initializePopulation()

    for j in range(500):
        
        eng.runGeneration(j)
    
    gpu_time = open('shuttleGPU/shuttle_gpu_time', 'a+')    
    gpu_time.write(str(time.time() - start_time) + "\n")
    gpu_time.close()
    
    # Perform validation with champion.
    stats = eng.getTeamStats(0)
    
    gpu_champion = open('shuttleGPU/shuttle_gpu_champion', 'a+')
    for j in range(len(stats['validation'])):
        gpu_champion.write(str(stats['validation'][j]) + ', ')
    gpu_champion.write("\n")
    gpu_champion.close()
    
    gpu_learners = open('shuttleGPU/shuttle_gpu_learners', 'a+')
    gpu_learners.write(str(stats['numLearners']) + "\n")
    gpu_learners.close()
    
    gpu_features = open('shuttleGPU/shuttle_gpu_features', 'a+')
    for j in range(len(stats['numFeatures'])):
        gpu_features.write(str(stats['numFeatures'][j]) + ", ")
    gpu_features.write("\n")
    gpu_features.close()
    
    gpu_nodes = open('shuttleGPU/shuttle_gpu_nodes', 'a+')
    for j in range(len(stats['numNodes'])):
         gpu_nodes.write(str(stats['numNodes'][j]) + ", ")
    gpu_nodes.write("\n")       
    gpu_nodes.close()
    
derp = derp

 # With CPU
for i in range(30):

    start_time = time.time()

    try:
        os.makedirs('shuttleCPU/' + str(i))
    except OSError as error:
        print(error)

    eng = sbb.SBBEngine(200, actions, trainX=train[:,:-1], testX=test[:,:-1], trainY=train[:,-1], testY=test[:,-1], outDirectory = 'shuttleCPU/' + str(i),  recordPerformance = True, device='/CPU:0', seed=42 + i, pRemoval=0.5, pAddition=0.95, numElites = 200)
    eng.initializePopulation()

    for j in range(500):
        
        eng.runGeneration(j)
        
    cpu_time = open('shuttleCPU/shuttle_cpu_time', 'a+')    
    cpu_time.write(str(time.time() - start_time) + "\n")
    cpu_time.close()
    
    cpu_champion = open('shuttleCPU/shuttle_cpu_champion', 'a+')
    
    for j in range(len(stats['validation'])):
        cpu_champion.write(str(stats['validation'][j]) + ', ')
    cpu_champion.write("\n")
    cpu_champion.close()
    
    cpu_learners = open('shuttleCPU/shuttle_cpu_learners', 'a+')
    cpu_learners.write(str(stats['numLearners']) + "\n")
    cpu_learners.write("\n")
    cpu_learners.close()
    
    cpu_features = open('shuttleCPU/shuttle_cpu_features', 'a+')
    for j in range(len(stats['numFeatures'])):
        cpu_features.write(str(stats['numFeatures'][j]) + "\n")
    cpu_features.write("\n")
    cpu_features.close()
    
    cpu_nodes = open('shuttleCPU/shuttle_cpu_nodes', 'a+')
    for j in range(len(stats['numNodes'])):
         cpu_nodes.write(str(stats['numNodes'][j]) + "\n")
    cpu_nodes.write("\n")     
    cpu_features.close()