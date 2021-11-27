import numpy as np
import tensorflow as tf
from SBBGP import SBBEngine as sbb


train = np.loadtxt('data/shuttletrain.txt')
test = np.loadtxt('data/shuttletest.txt')

train = tf.convert_to_tensor(train)
test = tf.convert_to_tensor(test)

sbb.