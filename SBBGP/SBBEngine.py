import numpy as np
import tensorflow as tf
import copy
import time
import os

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
                minInitTeamSize=3, minTeamSize=3, maxTeamSize=80, pAddition=0.9, pMutation=0.9, pRemoval=0.9, 
                sampling=None, sampleSize=2000, trainX=None, testX=None, trainY=None, testY=None, recordPerformance=False,
                outDirectory=None, maxTreeDepth = 8, device="/GPU:0", seed=0, numElites = 10, shouldPrecompute = True):
        
        # Parameters for controlling evolution.
        self.minTeamSize = minTeamSize
        self.maxTeamSize = maxTeamSize
        self.pAddition = pAddition
        self.pMutation = pMutation
        self.pRemoval = pRemoval
        self.popSize = popSize
        self.maxTreeDepth = 8
        self.gap = gap
        self.minInitTeamSize = minInitTeamSize
        self.maxInitTeamSize = maxInitTeamSize
        
        # For defining the sampling strategy.
        if sampling == 'classBalanced':
            self.sampling = sampling
            self.sampleSize = sampleSize
        else:
            self.sampling = None
            self.sampleSize = -1
            
        # Guides recordkeeping
        self.outDirectory = outDirectory
        self.recordPerformance = recordPerformance
        self.numElites = numElites
        
        # Performance (?).
        self.shouldPrecompute = shouldPrecompute
        self.precomputed = []
        
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
        
        # Call to initialize the tensorGP engine.
        self.engine = engine.Engine(fitness_func = self.dummy_fit, target_dims = [trainX.shape[0], 1], terminal_set=self.termSetTrain, effective_dims = 1, max_tree_depth = maxTreeDepth, device="/GPU:0", seed=seed)
        engine.set_device(device)
        
        self.actions = copy.deepcopy(actions)
    
    def initializePopulation(self):
        # Decide how large each team will be initially.
        teamSizes = np.random.randint(self.minInitTeamSize, self.maxInitTeamSize, self.popSize)
        self.totalPrograms = teamSizes.sum()
        
        offset = 0
           
        self.teams = []
           
        # Create initial teams and assign learners.
        for i in range(self.popSize):
            self.teams.append(Team(np.arange(offset, offset + teamSizes[i])))
            offset += teamSizes[i]
           
        # Call to initialize the TensorGP engine population. 
        self.engine.population, _ = self.engine.initialize_population(self.engine.max_init_depth, self.engine.min_init_depth, self.totalPrograms, self.engine.method, self.engine.max_nodes)
        
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
        
        tensors = tf.convert_to_tensor(self.engine.calculate_tensors(newPop)[0])
            
        winningBids = np.array(tf.argmax(tf.transpose(tensors), axis=2)[0])
        
        winningLearners = team.learners[winningBids]
        
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
    def teamSharedFitness(self, tensors, labels):
        
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
        
        # Cast to 32-bit float for future computations
        correct = tf.cast(correct, tf.float32)
        
        # Determine competitive fitness sharing coefficients. The second line is to avoid divide by 0 which would cause headache.
        # Those coefficients being 1 doesn't affect end results, because definitionally no team can gain fitness from an sample if
        # nobody got it right.
        correctSum = tf.math.count_nonzero(correct, axis=1, dtype=tf.float32)
        correctSum = tf.where(tf.equal(correctSum, 0), tf.ones_like(correctSum), correctSum)
        coefficients = 1 / correctSum
        coefficients = tf.reshape(coefficients, (-1, 1))

        CBAs = np.zeros((len(self.actions), self.popSize))
        
        # Initialize fitness for all teams to 0.
        fitnesses = tf.zeros(self.popSize)
        nonSharedFitness = tf.zeros(self.popSize)
        
        for i in range(len(self.actions)):
            
            # Find indicies of class i
            currClassIndices = tf.reshape(tf.where(labels == self.actions[i]), -1)
        
            # Compute base fitness as the correct answers times their coefficient.
            baseFitness = tf.math.reduce_sum(tf.gather(coefficients, currClassIndices) * tf.gather(correct, currClassIndices), axis=0)
        
            # Fitness without sharing.
            if self.recordPerformance:
                CBAs[i] = tf.math.reduce_sum(tf.gather(correct, currClassIndices), axis=0) / len(currClassIndices)
        
            # Divide by class size for actual fitness.
            fitnesses += baseFitness / len(currClassIndices)       

        nonSharedFitness = np.sum(CBAs, axis=0)
        
        # Assign fitness.
        for i in np.nditer(np.arange(len(self.teams))):

            # Fitness assigned as total number of correct answers.
            self.teams[i].fitness = fitnesses[i]
            self.teams[i].CBA = CBAs[:,i]

        return CBAs, fitnesses, nonSharedFitness
    
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
    def getTeamStats(self, teamRank, isElite = False):
    
        team = self.teams[teamRank]
    
        accuracy = -1
        cfMatrix = np.zeros((len(self.actions), len(self.actions)))
        precision = np.zeros(len(self.actions)) - 1
        recall = np.zeros(len(self.actions)) - 1
    
        if isElite:
            validationOutput, nonHitchhikers = self.predictTeamWithHitchhikers(team, self.termSetTest)

            # Compute confusion matrix.
            for i in range(len(self.actions)):
                predict = validationOutput == self.actions[i]
                for j in range(len(self.actions)):
                    truth = self.testY == self.actions[j]
                    cfMatrix[i][j] = np.sum(np.logical_and(predict, truth))
            
            # Use confusion matrix to compute precision, recall.
            for i in range(len(self.actions)):
                precisionDenominator = np.sum(cfMatrix[i,:])
                if precisionDenominator == 0:
                    precision[i] = 0
                else:
                    precision[i] = cfMatrix[i][i] / precisionDenominator
                    
                recallDenominator = np.sum(cfMatrix[:,i])
                if recallDenominator == 0:
                    recall[i] = 0
                else:
                    recall[i] = cfMatrix[i][i] / recallDenominator
        
            # Take the sum along the diagonal and divide it by |dataset|
            accuracy = np.trace(cfMatrix) / len(validationOutput)
        
        else:
            validationOutput = []
            nonHitchhikers = self.teams[teamRank].learners
    
        nodesList = np.zeros(len(nonHitchhikers))
        featuresList = [None for x in range(len(nonHitchhikers))]
            
        for i in range(len(nonHitchhikers)):
            nodesList[i], featuresList[i] = self.traverseTree(self.engine.population[nonHitchhikers[i]]['tree'])            
            
        numFeatures = [0 for featureSet in featuresList]
        
        for i in range(len(featuresList)):
            for feature in featuresList[i]:
                if 'scalar' not in feature:
                    numFeatures[i] += 1
        
        team.stats = {"validation":accuracy, "numLearners":len(team.learners), "numHitchhikers":len(team.learners) - len(nonHitchhikers), "numNodes":nodesList, "numFeatures":numFeatures,
            "recall":recall, "precision":precision, "hasEliteStats":isElite}
        
        return team.stats

    # Function to get all stats in order of rank?
    def getPopulationStats(self, teamRank):
        pass

    def runGeneration(self, gen):
        
        # Hold on to for recording stats at end of gen.
        oldPopSize = len(self.engine.population)
        
        print("Starting fitness calc")
        startTime = time.time()
        
        tensors = 0
        
        # if true: only compute using trees that have not been computed with yet.
        # if false: compute output of ALL trees. Much slower.
        if self.shouldPrecompute:
            tensors = self.engine.calculate_tensors(self.engine.population[len(self.precomputed):])
            tensors = self.precomputed + tensors[0]
            
        else:
            tensors = self.engine.calculate_tensors(self.engine.population)
            
        tensors = tf.convert_to_tensor(tensors)
        
        midTime = time.time()
        computeTime = midTime - startTime
        
        CBA, fitnesses, baseFitnesses = self.teamSharedFitness(tensors, self.trainY)
        CBA = [individual for _, individual in sorted(zip(fitnesses, CBA.tolist()))]
        self.rank()

        postTime = time.time() 
        assignTime = postTime - midTime

        # Need to take stats before axing teams.
        if self.recordPerformance:
        
            # Record stats on the whole population.
            for i in range(self.popSize):
                genDirectory = self.outDirectory + '/gen ' + str(gen) + ' stats'
                try:
                    os.makedirs(genDirectory)
                except OSError as error:
                    pass
            
                curTeam = self.teams[i]
            
                if curTeam.getStats():
                    # In case fitness sharing bumps a team up into "elite" status.
                    if i < (self.popSize * self.gap) and not curTeam.getStats()['hasEliteStats']: 
                        stats = self.getTeamStats(self.popSize - i - 1, isElite = True)
                    else:
                        stats = curTeam.getStats()
                else:
                    stats = self.getTeamStats(self.popSize - i - 1, isElite = (i < (self.popSize * self.gap)))
            
                file = open(genDirectory + '/stats.txt', 'a+')
                file.write('{\n')
                for key in stats.keys():
                    file.write(key + ": " + str(stats[key]) + "\n")
                file.write('}\n')
                file.close()
                
                file = open(genDirectory + '/CBAs.txt', 'a+')
                file.write('{\n')
                file.write(str(curTeam.CBA) + '\n')
                file.write('}\n')
                file.close()
                
                tagRank = [team.tag for team in self.teams[int(self.gap * self.popSize):]]
                tagRank.reverse()
                
                file = open(genDirectory + '/tag rank.txt', 'a+')
                file.write(str(tagRank)[1:-1])
                file.close()

        postEvalTime = time.time()

        # Purge unreferenced programs. First, find which ones are referenced in the surviving teams.
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
        
        if self.shouldPrecompute:
            self.precomputed = [tensors[program] for program in found]
        
        # Recreate population from found references.
        for i, program in enumerate(found):
            newPopulation.append(self.engine.population[program])
            newLearnerActions[i] = self.learnerActions[program]
        
        self.learnerActions = newLearnerActions
        self.engine.population = newPopulation
        
        # Addition mutation should pull from learners that existed from start of generation.
        # AND ARE OF THE TOP TEAMS!
        initialLearnerCount = self.totalPrograms
        
        # Determine probabilites that favor classes less represented in overall pop.
        # First, count # for each class.
        voteCounts = np.unique(self.learnerActions, return_counts=True)
        
        missing = []
        # Determine if there are NO occurences of a vote based on vote counts.
        for vote in self.actions:
            if vote not in voteCounts[0]:
                missing.append(vote)

        countsList = list(voteCounts[1])
        
        # FIGURE OUT WHY self.learnerActions IS FP!
        votesList = list(np.array(voteCounts[0], dtype=np.int32))
        
        votesList = votesList + missing
        countsList = countsList + [0 for i in range(len(missing))]

        countsArray = np.array(countsList)

        sum = np.sum(countsArray)
        
        # Apply a simple Laplace smooth to the counts
        smoothedCounts = countsArray + 1
        
        # Assign reverse probability scores.
        reversePoints = smoothedCounts / sum
        reverseProbabilities = reversePoints / np.sum(reversePoints)

        # Replace the gap% teams
        for i in range(0, int(self.gap * self.popSize)):
            
            # Parents for crossover.
            P1 = self.teams[np.random.randint(int(self.gap * self.popSize), high=self.popSize)]
            P2 = P1
            
            # Guaranteed to find a second unique parent, won't exit while loop until done.
            while P2 == P1:
                P2 = self.teams[np.random.randint(int(self.gap * self.popSize), high=self.popSize)]
            
            child = P1.oneChildCrossover(P2, self.learnerActions, self.maxTeamSize)
            
            if (len(self.engine.population) != len(self.learnerActions)):
                derp = derp
            
            child.addInd(self.maxTeamSize, self.pAddition, initialLearnerCount)
            child.removeInd(self.minTeamSize, self.pRemoval)
            newPrograms, newActions = child.mutateInd(self.pMutation, self.engine, self.learnerActions, self.actions, self.maxTreeDepth, reverseProbabilities, votesList)
            
            self.teams[i] = child
            
            # Adding any new programs and their actions to where they need to go.
            for i in range(len(newPrograms)):
                self.engine.population.append(newPrograms[i])
                self.learnerActions = np.append(self.learnerActions, newActions[i])
            
        # Legacy(?) code from when crossover produced two children.
        
        '''C1 = copy.deepcopy(P1)
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
            self.learnerActions = np.append(self.learnerActions, newActions[i])'''

        evoTime = time.time() - postTime

        if self.recordPerformance:
            
            file = open(self.outDirectory + '/pop.txt', 'a+')
            file.write(str(oldPopSize) + '\n')
            file.close()
            
            file = open(self.outDirectory + '/time.txt', 'a+')
            file.write(str(computeTime) + ', ' + str(assignTime) + ', ' + str(evoTime) + ', ' + str(postEvalTime - postTime) + '\n')
            file.close()
            
            minFitness = float(min(fitnesses))
            maxFitness = float(max(fitnesses))
            
            avgFitness = float(np.sum(fitnesses)) / len(self.teams)
            
            file = open(self.outDirectory + '/fitness.txt', 'a+')
            file.write(str(minFitness) + ',' + str(avgFitness) + ',' + str(maxFitness) +'\n')
            file.close()
            
            minBaseFitness = float(min(baseFitnesses))
            maxBaseFitness = float(max(baseFitnesses))
            
            avgBaseFitness = float(np.sum(baseFitnesses)) / len(self.teams)
            
            file = open(self.outDirectory + '/baseFitness.txt', 'a+')
            file.write(str(minBaseFitness) + ',' + str(avgBaseFitness) + ',' + str(maxBaseFitness) + '\n')
            file.close()
    
    # Used for trainSuperteams()
    def setTrainingSet():
        pass
    
    # One way to handle truly massive datasets - simply train a set of small teams nice and fast.
    # Use exact = True when wanting all data to be used to train one subteam. Number of subteams = |trainingSet| / n, rounded up
    # Use exact = False in conjunction with numMembers to decide to just train x teams.
    def trainSuperteam(self, trainingSetSource, n=500000, generations=50, exact=True, numMembers=False):
        
        eliteTeams = []
        elitePrograms = []
        
        self.initPopulation()
        
        # Split input training data into subsets s where |s| = n
        if exact:
            pass
        
        # Start training the next team.
        for i in range(nSubteams):
            
            self.initializePopulation()
            
            # Train the superteam.
            for j in range(generations):
                self.runGeneration(j)
            
        #Evaluate superteam.

# Object representing a SuperTeam, needed for SBBEngine.trainSuperteam()
class SuperTeam:
    
    def __init__(self):
        pass

# Object representing a team. Mostly just a list of '''references''' to programs in the loosest possible definition
class Team:
    
    nextTag = 0
    
    # Needs actual initialization logic.
    def __init__(self, learners):
        self.learners = learners
        self.tag = Team.nextTag
        Team.nextTag += 1
        fitness = 0
        self.stats = False
        self.CBA = None
        
    def getStats(self):
        return self.stats
        
    # Operator for removing a program for a team.
    def removeInd(self, minSize, pRemoval):
        b = np.random.rand()
        mask = np.ones(len(self.learners), dtype=bool)
        while b > (1 - pRemoval) and np.sum(mask) > minSize:
            mask[np.random.randint(len(self.learners))] = False
        
        self.learners = self.learners[mask]
    
    # Operator for adding an existing program in the pool to a team.
    def addInd(self, maxSize, pAddition, initialLearnerCount):
        b = np.random.rand()
        while b > (1 - pAddition) and len(self.learners) < maxSize:
            toAdd = np.random.randint(initialLearnerCount)
            if toAdd not in self.learners:
                self.learners = np.append(self.learners, toAdd)
                
            b = b * np.random.rand()
    
    # Crossover operation that returns a single child team from two parents.
    def oneChildCrossover(self, otherParent, actionsList, maxSize):
    
        other = []
        learners = np.array([], dtype=np.int32)
    
        for learner in self.learners:
            if learner in otherParent.learners:
                learners = np.append(learners, learner)
            else:
                other.append(learner)
    
        # Set of classes represented in team - used for deciding what programs to insert.
        represented = set({})
    
        for learner in learners:
            represented.add(actionsList[learner])
        
        done = False
        
        # Randomly select the 'other' programs for insertion into the new team.
        while not done:
        
            toRemove = []
        
            for i in range(len(other)):
                if actionsList[other[i]] in represented:
                    toRemove.append(i)
        
            other = list(set(other).difference(toRemove))
        
            if len(learners) >= maxSize or len(other) == 0:
                done = True
        
            else:
                nextToAdd = np.random.choice(other)
                learners = np.append(learners, nextToAdd)
                other.remove(nextToAdd)
                represented.add(actionsList[nextToAdd])
        
        return Team(learners)
    
    # Function that handles both mutation and crossover as an all-in-one package. Prime spaghetti to refactor.
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
                
                # Pick parents for crossover. Second parent must have same vote as first.
                P1 = self.learners[learner]
                possibleParents = learners[actionsList == actionsList[learner]]
                
                P2 = P1
                while P2 == P1:
                    P2 = self.learners([np.random.choice(possibleParents)])

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
        
    def mutateInd(self, pMutation, engine, actionsList, actions, maxTreeDepth, classProbabilities, classVotes):
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

                # Crossover and mutate.
                newProgram = copy.deepcopy(engine.population[P1])
                newProgram['tree'] = engine.mutation(newProgram['tree'])
                    
                depth, _ = newProgram['tree'].get_depth()
                #print(depth)
                    
            newPrograms.append(newProgram)
                
            notDone.pop(learnerIndex)
            
            if b < pMutation / 5:

                rolledClass = np.random.rand()
                
                sum = 0
                
                i = 0
                
                while rolledClass > sum:
                    sum += classProbabilities[i]
                    if rolledClass <= sum:
                        chosenClass = i
                    else:
                        i += 1

                newActions.append(actions[chosenClass])
                
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