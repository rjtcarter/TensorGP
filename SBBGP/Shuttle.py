import SBBEngine
import numpy as np
from sklearn.model_selection import train_test_split 

# Load data from file.
train_data = np.genfromtxt('data/shuttle.trn', delimiter=' ')
test_data = np.genfromtxt('data/shuttle.tst', delimiter=' ')

# Classification Problem: just need to get possible classes as actions.
actions = np.unique(train_data[:,-1])

SBBEngine.SBBEngine(200, actions, 

(self, popSize, actions, features, labels, gap, maxInitTeamSize,
                minInitTeamsSize, minTeamSize, maxTeamSize, pAddition, pMutation, pRemoval)