import numpy as np
import tensorflow as tf

# Mostly an empty shell.
class SBBEngine:

    # Teams is gonna be a list - for now, anyways.
    teams = []
    
    # Set of possible actions.
    possibleActions = []
    
    # actionTensor is a tensor describing the chosen action of each learner.
    actionTensor
    
    # True class labels.
    labels = []
    
    # Attributes
    features = []

    # Needs initalization logic.
    def __init__(self, popSize, actions, labels, features):
        for i in range(popSize):
            self.teams.append(Team())
        
        for i in range(len(actions))
            self.actions.append(actions[i])
            
        self.features = tf.tensor(features)
        self.labels = tf.tensor(labels)
       
    # Takes in the tensor and goes through the pop, determines classification accuracy
    # and applies as fitness.
    def teamFitness(self, programOutput):
        for i in np.nditer(np.arange(len(teams))):
        
            # Grab winning learner for each team
            winningBids = tf.argmax(programOutput.T * self.learners)
            
            # Getting the chosen actions from the selected bids.
            outcomes = actionList[winningBids]
            
            # correctness is a binary list - True where the chosen action was correct and false otherwise.
            correctness = tf.math.equals(output, labels)
            
            # Fitness assigned as total number of correct answers.
            teams[i].fitness = correctness.sum()

# Object representing a team.
class Team:
    
    # Needs actual initialization logic.
    def __init__(self, numLearners):
        self.learners = tf.zeros(numLearners)
        fitness = 0
        
    # Use to generate a new team randomly with certain parameters.
    def randTeam():
        raise NotImplementedError()
       
    # Given the output of the programs, compute what the team does for each exemplar.
    # ProgramOutput is program x exemplars.
    # outcomes is 1D, with # elements = # exemplars. Here in large part as legacy.
    def calcOutput(self, programOutput, actionList):
        winningBids = tf.argmax(programOutput.T * self.learners)
        outcomes = actionList[winningBids]
        return outcomes
        
# Relates what programs suggest what actions.
# Here as a class for sake of organization, though not absolutely necessary.
class ActionList:

    # Will need more creative means to assign this.
    def __init__(self, actions):
        self.actions = tf.tensor(actions)