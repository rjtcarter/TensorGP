import numpy as np
import tensorflow as tf
import copy

# Need to import tensorgp based on relative path.
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tensorgp'))
import engine

# A not so empty shell, now.
class SBBEngine:

    # Needs initalization logic.
    def __init__(self, popSize, actions, gap=0.5, maxInitTeamSize=6,
                minInitTeamsSize=3, minTeamSize=3, maxTeamSize=8, pAddition=0.9, pMutation=0.9, pRemoval=0.9, 
                sampling=None, sampleSize=2000, trainX=None, testX=None, trainY=None, testY=None):
        
        # Parameters for controlling mutation.
        self.minTeamSize = minTeamSize
        self.maxTeamSize = maxTeamSize
        self.pAddition = pAddition
        self.pMutation = pMutation
        self.pRemoval = pRemoval
        
        # For defining the sampling strategy.
        if sampling == 'classBalanced':
            self.sampling = sampling
            self.sampleSize = sampleSize
        else:
            self.sampling = None
            self.sampleSize = -1
            
        
        # Converting data to tensors, if it isn't already.
        
        
        # Converting to terminal sets, so TensorGP is happy.
        self.termSetX = Terminal_Set(1, trainX.shape[0])
        self.termSetY = Terminal_Set(1, trainY.shape[0])
        for i in range(trainX.shape[0]): 
            self.termSetX.add_to_set(str(i), trainX[:,i])
            self.termSetY.add_to_set('x', tf.zeros(res))
        
        # Decide how large each team will be initially.
        teamSizes = np.random.randint(minInitTeamsSize, maxInitTeamSize, popSize)
        self.totalPrograms = teamSizes.sum()
        
        offset = 0
           
        # Create initial teams and assign learners.
        for i in range(popSize):
            self.teams.append(Team([ind for j in range(offset + teamSizes[i])]))
            offset += teamSizes[i]
           
        # Call to initialize the TensorGP engine. 
        self.engine = Engine(target_dims = [trainX.shape[0], 1], terminal_set=termSetX, effective_dims = 1)
        self.engine.initialize_population(self.engine.max_init_depth, self.engine.min_init_depth, self.totalPrograms, self.engine.method, self.engine.max_nodes)
        
        self.gap = gap
        
        self.actions = copy.deepcopy(actions)
        
        # Relates what action a learner does to the learner.
        self.learnerActions = np.random.choice(self.actions, size = self.totalPrograms)
       
       
       
    # Rankings determined by fitness, sort from least fit to most.
    def rank(self):
        self.teams.sort(key=lambda team: team.fitness, reverse=True)

    
       
    # Takes in the tensor and goes through the pop, determines classification accuracy
    # and applies as fitness.
    def teamFitness(self, programOutput):
        # iterate over teams.
        for i in np.nditer(np.arange(len(teams))):
        
            # Grab winning learner for each team
            winningBids = teams[i].learners[tf.argmax(tf.gather(programOutput.T, teams[i].learners))]
            
            # Getting the chosen actions from the selected bids.
            outcomes = learnerActions[winningBids]
            
            # correctness is a binary list - True where the chosen action was correct and false otherwise.
            correctness = tf.math.equals(output, labels)
            
            # Fitness assigned as total number of correct answers.
            teams[i].fitness = correctness.sum()

    def runGeneration(self):
        
        # Measure fitness and rank
        self.teamFitness(self.engine.calculate_tensors(engine.population))
        self.rank()
        
        # Addition mutation should pull from learners that existed from start of generation.
        initialLearnerCount = self.totalPrograms
        
        # Replace the gap% individuals
        for i in range(int(gap * popSize)):
            # Huzzah, mutation. First, picking which team will be the basis of the new guy.
            self.teams[i] = copy.deepcopy(self.teams[np.random.randint(int(gap * popSize), high=popSize)])
            
            # Then, the actual mutation:
            self.teams[i].addInd(self.maxTeamSize, self.pAddition, initialLearnerCount)
            self.teams[i].removeInd(self.minTeamSize, self.pRemoval)
            newPrograms, newActions = self.teams[i].mutateInd(self.pMutation, len(self.actions), self.engine)
            
            # Adding any new programs and their actions to where they need to go.
            for i in range(len(newPrograms)):
                engine.population.append(newPrograms[i])
                learnerActions.append(newActions[i])
            
            
        # Purge unreferenced programs. First, find which ones are referenced.
        found = []
        for i in range(len(self.teams)):
            for j in range(len(self.teams[i])):
                if teams[i].learners[j] not in found:
                    found.append(teams[i].learners[j])
                # Adjust references.
                teams[i].learners[j] = found.index(teams[i].learners[j])
        
        # Recreate population from found references.
        for i in range(len(found)):
            newPopulation.append(engine.population[i])
        
        engine.population = newPopulation
        

# Object representing a team.
class Team:
    
    # Needs actual initialization logic.
    def __init__(self, learners):
        self.learners = learners
        fitness = 0
        
    def removeInd(self, minSize, pRemoval):
        b = 1
        while b > (1 - pRemoval) and len(self.learners) > minSize:
            self.learners.pop(np.random.randint(len(self.learners)))
            b = b * np.random.rand()
    
    def addInd(self, maxSize, pAddition, initialLearnerCount):
        b = 1
        while b > (1 - pAddition) and len(self.learners) < maxSize:
            toAdd = np.random.randint(initialLearnerCount)
            if toAdd not in self.learners:
                self.learners.append(toAdd)
                
            b = b * np.random.rand()
    
    def mutateInd(self, pMutation, possibleActions, engine):
        newActions = []
        newPrograms = []
        while len(newPrograms) == 0:
            for learner in self.learners:
                if np.random.rand() < pMutation:
                    newPrograms.append(copy.deepcopy(engine.population[learner]))
                    newPrograms[-1] = engine.mutation(newPrograms[-1])
                    newActions.append(np.random.randint(possibleActions))
        return newPrograms, newActions
        
        
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