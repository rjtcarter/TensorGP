import numpy as np
import tensorflow as tf
import copy
import time

# Need to import tensorgp based on relative path.
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tensorgp'))
import engine

# A not so empty shell, now.
class SBBEngine:

    # Need to provide a fitness function for it to apply to trees - but I don't want it really doing *anything*
    # This is a dummy intended to sate TensorGP's lust and do nothing more.
    def dummy_fit(self, **kwargs):
        population = kwargs.get('population')
        tensors = kwargs.get('tensors')
        
        best_ind = 0

        for i in range(len(tensors)):
            population[i]['fitness'] = 0

        return population, population[best_ind]


    # Initialization of SBB.
    def __init__(self, popSize, actions, gap=0.5, maxInitTeamSize=6,
                minInitTeamsSize=3, minTeamSize=3, maxTeamSize=80, pAddition=0.9, pMutation=0.9, pRemoval=0.9, 
                sampling=None, sampleSize=2000, trainX=None, testX=None, trainY=None, testY=None, recordPerformance=False,
                outDirectory=None, maxTreeDepth = 8, device="/GPU:0", seed=0):
        
        # Parameters for controlling mutation.
        self.minTeamSize = minTeamSize
        self.maxTeamSize = maxTeamSize
        self.pAddition = pAddition
        self.pMutation = pMutation
        self.pRemoval = pRemoval
        self.popSize = popSize
        self.maxTreeDepth = 8
        
        # For defining the sampling strategy.
        if sampling == 'classBalanced':
            self.sampling = sampling
            self.sampleSize = sampleSize
        else:
            self.sampling = None
            self.sampleSize = -1
            
        
        self.outDirectory = outDirectory
        self.recordPerformance = recordPerformance
        
        try:
            os.mkdir(outDirectory)
        except:
            pass
        
        # Converting to terminal sets, so TensorGP is happy.
        self.termSetTrain = engine.Terminal_Set(1, [trainX.shape[0], 1])
        for i in range(trainX.shape[1]): 
            self.termSetTrain.add_to_set(str(i), trainX[:,i])
        
        self.trueTestLength = len(testX) 
        
        testX = np.pad(testX, ((0, trainX.shape[0] - testX.shape[0]),(0, 0),(0, 0)))
        testX = tf.convert_to_tensor(testX, dtype=tf.float32)
        
        self.termSetTest = engine.Terminal_Set(1, [testX.shape[0], 1]) 
        for i in range(testX.shape[1]):
            self.termSetTest.add_to_set(str(i), testX[:,i])
        
        self.trainY = trainY[:,0]
        self.testY = testY[:,0]
        
        # Decide how large each team will be initially.
        teamSizes = np.random.randint(minInitTeamsSize, maxInitTeamSize, popSize)
        self.totalPrograms = teamSizes.sum()
        
        offset = 0
           
        self.teams = []
           
        # Create initial teams and assign learners.
        for i in range(popSize):
            self.teams.append(Team(np.arange(offset, offset +teamSizes[i])))
            offset += teamSizes[i]
           
        # Call to initialize the TensorGP engine. 
        self.engine = engine.Engine(fitness_func = self.dummy_fit, target_dims = [trainX.shape[0], 1], terminal_set=self.termSetTrain, effective_dims = 1, max_tree_depth = maxTreeDepth, device="/GPU:0", seed=seed)
        self.engine.population, _ = self.engine.initialize_population(self.engine.max_init_depth, self.engine.min_init_depth, self.totalPrograms, self.engine.method, self.engine.max_nodes)
        engine.set_device(device)
        
        self.gap = gap
        
        self.actions = copy.deepcopy(actions)
        
        # Relates what action a learner does to the learner.
        self.learnerActions = np.random.choice(self.actions, size = self.totalPrograms)
     
    # Use this for evaluating a specific team.
    def evaluateTeam(self, team, data, labels):
    
        start = time.time()
    
        outcomes = self.predictTeam(team, data)
       
        correctness = tf.cast(tf.math.equal(outcomes[:len(labels)], labels), tf.int32)
        
        accuracy = tf.reduce_sum(correctness) / len(correctness)
        
        return accuracy
        
    def predictTeam(self, team, data):
    
        newPop = []
        
        actualTermSet = self.engine.terminal
        
        if not isinstance(data, engine.Terminal_Set):
            termSet = engine.Terminal_Set(1, [data.shape[0], 1])
            for i in range(data.shape[1]): 
                termSet.add_to_set(str(i), self.trainX[:,i])
            
            self.engine.terminal = termSet
            
        else:
            self.engine.terminal = data
        
        for learner in team.learners:
            newPop.append(self.engine.population[learner])
        
        tensors = tf.convert_to_tensor(self.engine.calculate_tensors(newPop)[0])
            
        winningBids = team.learners[np.array(tf.argmax(tf.transpose(tensors), axis=2))[0]]
        
        outcomes = self.learnerActions[winningBids]
        
        outcomes = outcomes[:self.trueTestLength]
        
        self.engine.terminal = actualTermSet
    
        return outcomes
        
    def predictTeamWithHitchhikers(self, team, data):
    
        newPop = []
        
        actualTermSet = self.engine.terminal
        
        if not isinstance(data, engine.Terminal_Set):
            termSet = engine.Terminal_Set(1, [data.shape[0], 1])
            for i in range(data.shape[1]): 
                termSet.add_to_set(str(i), self.trainX[:,i])
            
            self.engine.terminal = termSet
            
        else:
            self.engine.terminal = data
        
        for learner in team.learners:
            newPop.append(self.engine.population[learner])
        
        print(team.learners)
        
        tensors = tf.convert_to_tensor(self.engine.calculate_tensors(newPop)[0])
            
        winningBids = np.array(tf.argmax(tf.transpose(tensors), axis=2)[0])
        
        winningLearners = team.learners[winningBids]
        
        print(winningBids)
        
        nonHitchhikers = np.unique(winningBids)
        
        outcomes = self.learnerActions[winningLearners]
        
        outcomes = outcomes[:self.trueTestLength]
        
        self.engine.terminal = actualTermSet
    
        return outcomes, nonHitchhikers
       
    # Rankings determined by fitness, sort from least fit to most.
    def rank(self):
        self.teams.sort(key=lambda team: team.fitness)

    # Takes in the tensor and goes through the pop, determines classification accuracy
    # and applies as fitness.
    def teamFitness(self, programOutput, labels):

        start = time.time()

        tensors = tf.convert_to_tensor(programOutput[0])
        allOutcomesList = []
        
        '''learnersList = np.zeros((self.maxTeamLength, len(self.teams))) - 1
        for i in len(self.teams):
            learnersList[i][:len(self.teams[i].learners)] = self.teams[i].learners
        
        learnersTensor = tf.convert_to_tensor(learnersList)
        
        winningBids = learnersTensor.gather((tf.argmax(tf.gather(tf.transpose(tensors), learnersTensor, axis=2), axis=2)'''
        
        # iterate over teams.
        '''for i in np.nditer(np.arange(len(self.teams))):
        
            # Grab winning learner for each team
            winningBids = self.teams[i].learners[np.array(tf.argmax(tf.gather(tf.transpose(tensors), self.teams[i].learners, axis=2), axis=2)[0])]
            
            # Getting the chosen actions from the selected bids.
            outcomes = self.learnerActions[winningBids]
            
            # ADD TO TENSOR OF ALL OUTCOMES
            allOutcomesList.append(outcomes) 
            
            # correctness is a binary list - True where the chosen action was correct and false otherwise.
            correctness = tf.cast(tf.math.equal(outcomes, labels), tf.int32)

            # Fitness assigned as total number of correct answers.
            self.teams[i].fitness = tf.reduce_sum(correctness)'''

        allOutcomes = tf.stack(allOutcomesList)

        return allOutcomes
        
    # Class balanced fitness function that shares fitness based on 'novelty'.
    def teamSharedFitness(self, programOutput, labels):
        
        x = time.time()

        # Convert tree output to usablef form.
        tensors = tf.convert_to_tensor(programOutput[0])
        
        # List of all teams' outputs.
        allOutcomesList = []
        
        # iterate over teams.
        for i in np.nditer(np.arange(len(self.teams))):
        
            # Grab winning learner for each team
            winningBids = self.teams[i].learners[np.array(tf.argmax(tf.gather(tf.transpose(tensors), self.teams[i].learners, axis=2), axis=2)[0])]
            
            # Getting the chosen actions from the selected bids.
            outcomes = self.learnerActions[winningBids]
            
            # ADD TO LIST OF ALL OUTCOMES
            allOutcomesList.append(outcomes)

        # Convert list to tensor.
        allOutcomesTensor = tf.cast(tf.stack(allOutcomesList, axis=1), dtype=tf.float32)
        
        # Determine all correct answers.
        correct = (allOutcomesTensor == tf.reshape(labels, (-1, 1)))
        
        #print(correct)
        
        # Cast to 32-bit float for future computations
        correct = tf.cast(correct, tf.float32)
        
        #print("CORRECTNESS AFTER INT CASTING")
        #print(correct)
        
        # Determine competitive fitness sharing coefficients.
        coefficients = 1 / tf.math.count_nonzero(correct, axis=1, dtype=tf.float32)
        coefficients = tf.reshape(coefficients, (-1, 1))
        
        #print(coefficients)
        
        CBA = tf.zeros(len(self.teams))
        
        # Initialize fitness for all teams to 0.
        fitnesses = tf.zeros(len(self.teams))
        
        for i in range(len(self.actions)):
            
            # Find indicies of class i
            currClassIndices = tf.reshape(tf.where(labels == self.actions[i]), -1)
        
            #print(tf.gather(coefficients, currClassIndices))
            #print(tf.gather(correct, currClassIndices))
        
            #print(tf.gather(coefficients, currClassIndices) * tf.gather(correct, currClassIndices))
        
            # Compute base fitness as the correct answers times their coefficient.
            baseFitness = tf.math.reduce_sum(tf.gather(coefficients, currClassIndices) * tf.gather(correct, currClassIndices), axis=0)
        
            #print(baseFitness)
        
            # Divide by class size for actual fitness.
            fitnesses += baseFitness / len(currClassIndices)
            CBA += tf.math.reduce_sum(tf.gather(correct, currClassIndices), axis=0) / len(currClassIndices)
        
        # Assign fitness.
        for i in np.nditer(np.arange(len(self.teams))):

            # Fitness assigned as total number of correct answers.
            self.teams[i].fitness = fitnesses[i]

        return CBA
    
    # Traverse a tree and return number of nodes + number of features.
    def traverseTree(self, tree):
        
        if tree.terminal:
            return 1, set({tree.get_str()})
        
        else:
        
            numNodes = 1
            featureSet = set({})
            
            for child in tree.children:
                tmp1, tmp2 = self.traverseTree(child)
                numNodes += tmp1
                featureSet = featureSet.union(tmp2)
            
            return numNodes, featureSet
    
    
    # Return validation performance, # learners, # of features each learner has, # of nodes each learner has  
    def getTeamStats(self, teamRank):
    
        team = self.teams[teamRank]
    
        validationOutput, nonHitchhikers = self.predictTeamWithHitchhikers(team, self.termSetTest)
    
        correctness = (validationOutput == self.testY)
        correctness = tf.cast(correctness, tf.float32)
    
        classAccuracy = np.zeros(len(self.actions))
    
        for i in range(len(self.actions)):
            currClassIndices = tf.reshape(tf.where(self.testY == self.actions[i]), -1)
            
            classAccuracy[i] = tf.math.reduce_sum(tf.gather(correctness, currClassIndices)) / len(currClassIndices)
            
        nodesList = np.zeros(len(nonHitchhikers))
        featuresList = [None for x in range(len(nonHitchhikers))]
            
        for i in range(len(nonHitchhikers)):
            nodesList[i], featuresList[i] = self.traverseTree(self.engine.population[nonHitchhikers[i]]['tree'])            
            
        numFeatures = [0 for featureSet in featuresList]
        
        for i in range(len(featuresList)):
            for feature in featuresList[i]:
                if 'scalar' not in feature:
                    numFeatures[i] += 1
        
        return {"validation":classAccuracy, "numLearners":len(team.learners), "numNodes":nodesList, "numFeatures":numFeatures}

    def runGeneration(self):
        
        print(len(self.engine.population))
        
        # Measure fitness and rank
        print("Starting fitness calc")
        start_time = time.time()
        tensors = self.engine.calculate_tensors(self.engine.population)
        print("Done computing tensors after: " + str(time.time() - start_time))
        mid_time = time.time()
        CBA = self.teamSharedFitness(tensors, self.trainY)
        print("Done assigning fitness after an additional: " + str(time.time() - mid_time))
        self.rank()
        
        maxVal = -1
        sum = 0
        
        '''for i in range(len(self.engine.population)):
            newVal, _ = self.traverseTree(self.engine.population[i]['tree'])
            if newVal > maxVal:
                maxVal = newVal
                
            sum += newVal

        print("Max tree size: " + str(maxVal))
        print("Average tree size: " + str(sum / len(self.engine.population)))'''

        # If writing to file...
        start = time.time()

        if self.recordPerformance:
            file = open(self.outDirectory + '/test.txt' , 'a+')
            file.write(str(float(self.evaluateTeam(self.teams[-1], self.termSetTest, self.testY))) + '\n')
            file.close()
            
            minFitness = float(min(CBA))
            maxFitness = float(max(CBA))
            
            avg = float(np.sum(CBA)) / len(self.teams)
            
            file = open(self.outDirectory + '/fitness.txt', 'a+')
            file.write(str(minFitness) + ',' + str(avg) + ',' + str(maxFitness) +'\n')
        
        # Addition mutation should pull from learners that existed from start of generation.
        initialLearnerCount = self.totalPrograms
        
        # Replace the gap% teams
        for i in range(0, int(self.gap * self.popSize), 2):
            
            # Parents for crossover.
            P1 = self.teams[np.random.randint(int(self.gap * self.popSize), high=self.popSize)]
            P2 = P1
            
            while P2 == P1:
                P2 = self.teams[np.random.randint(int(self.gap * self.popSize), high=self.popSize)]
            
            C1 = copy.deepcopy(P1)
            C2 = copy.deepcopy(P2)
            
            # Learners common to both parents are placed in both children.
            # Need to find the intersection, and those not in it.
            intersect = []
            other = []
            
            for learner in C1.learners:
                if learner in C2.learners:
                    intersect.append(learner)
                else:
                    other.append(learner)
                    
            for learner in C2.learners:
                if learner not in C1.learners:
                    other.append(learner)
            
            # Setting new team's learners to the intersection.
            C1.learners = copy.deepcopy(intersect)
            C2.learners = copy.deepcopy(intersect)
            
            self.teams[i] = C1
            self.teams[i] = C2
            
            # Dividing up rest of the learners.
            for learner in other:
                if len(C1.learners) <= len(C2.learners) and len(C1.learners) < self.minTeamSize:
                    C1.learners.append(learner)
                
                elif len(C2.learners) < self.minTeamSize:
                    C2.learners.append(learner)
                
                # Each team has at least min number of learners
                else:
                    p = np.random.rand()
                    
                    if p < 0.5:
                        C1.learners.append(learner)
                    else:
                        C2.learners.append(learner)
            
            C1.learners = np.array(C1.learners)
            C2.learners = np.array(C2.learners)
            
            # Then, the actual mutation:
            C1.addInd(self.maxTeamSize, self.pAddition, initialLearnerCount)
            C1.removeInd(self.minTeamSize, self.pRemoval)
            
            C2.addInd(self.maxTeamSize, self.pAddition, initialLearnerCount)
            C2.removeInd(self.minTeamSize, self.pRemoval)
            
            newPrograms, newActions = C1.mutateInd(self.pMutation, self.actions, self.engine, self.learnerActions, self.maxTreeDepth)
            
            # Adding any new programs and their actions to where they need to go.
            for i in range(len(newPrograms)):
                self.engine.population.append(newPrograms[i])
                self.learnerActions = np.append(self.learnerActions, newActions[i])
            
            newPrograms, newActions = C2.mutateInd(self.pMutation, self.actions, self.engine, self.learnerActions, self.maxTreeDepth)
            
            # Adding any new programs and their actions to where they need to go.
            for i in range(len(newPrograms)):
                self.engine.population.append(newPrograms[i])
                self.learnerActions = np.append(self.learnerActions, newActions[i])
            
            
            
        # Purge unreferenced programs. First, find which ones are referenced.
        found = []
        for i in range(len(self.teams)):
            for j in range(len(self.teams[i].learners)):
                if self.teams[i].learners[j] not in found:
                    found.append(self.teams[i].learners[j])
                # Adjust references.
                self.teams[i].learners[j] = found.index(self.teams[i].learners[j])

        self.totalPrograms = len(found)
        
                
        
        newPopulation = []
        newLearnerActions = np.zeros(len(found))
        
        # Recreate population from found references.
        for i, program in enumerate(found):
            newPopulation.append(self.engine.population[program])
            newLearnerActions[i] = self.learnerActions[program]
        
        self.learnerActions = newLearnerActions
        self.engine.population = newPopulation
        
        

# Object representing a team.
class Team:
    
    # Needs actual initialization logic.
    def __init__(self, learners):
        self.learners = learners
        fitness = 0
        
    def removeInd(self, minSize, pRemoval):
        b = np.random.rand()
        mask = np.ones(len(self.learners), dtype=bool)
        while b > (1 - pRemoval) and np.sum(mask) > minSize:
            mask[np.random.randint(len(self.learners))] = False
        
        self.learners = self.learners[mask]
    
    def addInd(self, maxSize, pAddition, initialLearnerCount):
        b = np.random.rand()
        while b > (1 - pAddition) and len(self.learners) < maxSize:
            toAdd = np.random.randint(initialLearnerCount)
            if toAdd not in self.learners:
                self.learners = np.append(self.learners, toAdd)
                
            b = b * np.random.rand()
    
    def mutateIndCrossover(self, pMutation, possibleActions, engine, actionsList, maxTreeDepth):
        newActions = []
        newPrograms = []
        counter = 0
        b = np.random.rand()
        notDone = list(range(len(self.learners)))
        
        while b < pMutation and len(notDone) > 0:
        

            depth = float('inf')
            
            while depth > maxTreeDepth:
        
                # pick the lucky learner
                learnerIndex = np.random.randint(len(notDone))
                learner = notDone[learnerIndex]
        
                # Pick parents for crossover.
                P1 = self.learners[learner]
                P2 = P1
                while P2 == P1:
                    P2 = np.random.randint(len(engine.population))

                #print(self.learners)
                #print(notDone)
                #print(learner)
                #print(P1)

                # Crossover and mutate.
                newProgram = copy.deepcopy(engine.population[P1])
                newProgram['tree'] = engine.crossover(engine.population[P1]['tree'], engine.population[P2]['tree'])
                newProgram['tree'] = engine.mutation(newProgram['tree'])
                    
                depth, _ = newProgram['tree'].get_depth()
                #print(depth)
                    
            newPrograms.append(newProgram)
                
            notDone.pop(learnerIndex)
            
            if b < pMutation / 5:
                newActions.append(possibleActions[np.random.randint(len(possibleActions))])
            else:
                newActions.append(actionsList[self.learners[learner]])
            
            self.learners[learner] = len(engine.population) - 1 + counter

            counter += 1
            b = np.random.rand()

        return newPrograms, newActions
        
    def mutateInd(self, pMutation, possibleActions, engine, actionsList, maxTreeDepth):
        newActions = []
        newPrograms = []
        counter = 0
        b = np.random.rand()
        notDone = list(range(len(self.learners)))
        
        while b < pMutation and len(notDone) > 0:
        

            depth = float('inf')
            
            while depth > maxTreeDepth:
        
                # pick the lucky learner
                learnerIndex = np.random.randint(len(notDone))
                learner = notDone[learnerIndex]
        
                # Pick parents for crossover.
                P1 = self.learners[learner]
               
                #print(notDone)
                #print(learner)
                #print(P1)

                # Crossover and mutate.
                newProgram = copy.deepcopy(engine.population[P1])
                newProgram['tree'] = engine.mutation(newProgram['tree'])
                    
                depth, _ = newProgram['tree'].get_depth()
                #print(depth)
                    
            newPrograms.append(newProgram)
                
            notDone.pop(learnerIndex)
            
            if b < pMutation / 5:
                newActions.append(possibleActions[np.random.randint(len(possibleActions))])
            else:
                newActions.append(actionsList[self.learners[learner]])
            
            self.learners[learner] = len(engine.population) - 1 + counter

            counter += 1
            b = np.random.rand()

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