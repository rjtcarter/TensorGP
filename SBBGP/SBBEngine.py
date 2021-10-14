import numpy as np
import tensorflow as tf

# Mostly an empty shell.
class SBBEngine:

    # Teams is gonna be a list - for now, anyways.
    teams = []
    
    # Set of possible actions.
    possibleActions = []
    
    # actionTensor is a tensor describing the chosen action of each learner.
    actionTensor = tf.zeros(1)
    
    # True class labels.
    labels = []
    
    # Attributes
    features = []


    # Needs initalization logic.
    def __init__(self, popSize, actions, labels, features, gap, maxInitTeamSize
                minInitTeamsSize):
        
        
        
        # Decide how large each team will be initially.
        teamSizes = np.random.randint(minInitTeamsSize, maxInitTeamSize, popSize)
        totalInitPrograms = teamSizes.sum()
        
        offset = 0
           
        # Create initial teams and assign learners.
        for i in range(popSize):
            self.teams.append(Team(totalInitPrograms()))
            self.teams[-1].learners = np.zeros(popSize)
            self.teams[-1].learners[offset:offset + teamSizes[i] + 1] = 1
            offset += teamSizes[i]
           
        # Call to initialize the TensorGP engine. 
        self.engine = Engine()
        
        self.gap = gap
        
        
        for i in range(len(actions))
            self.actions.append(actions[i])
            
        self.features = tf.tensor(features)
        self.labels = tf.tensor(labels)
       
    # Rankings determined by fitness, sort from least fit to most.
    def rank(self):
        self.teams.sort(key=lambda team: team.fitness, reverse=True)

    
       
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

    def runGeneration(self):
        
        # Measure fitness and rank
        self.teamFitness(self.engine.calculate_tensors(engine.population))
        self.rank()
        
        # Replace the gap% individuals
        for i in range(int(gap * popSize)):
            # TODO: create actual logic for replacing with a meaningful team.
            # Basically, need to handle variation operations for this to be meaningful
            self.teams[i] = 0
        
        
        
        
        
        
        

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