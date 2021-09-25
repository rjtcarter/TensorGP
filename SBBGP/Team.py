# Mostly an empty shell.
class SBBEngine:

    teams = []
    actionList = []
    labels = []
    features = []

    # Needs initalization logic
    def __init__(self, popSize, actions):
        for i in range(popSize):
            teams.append(Team())
        
        for i in range(len(actions))
            actionList.append(actions[i])
       
    # Takes in the tensor and goes through the pop, determines classification accuracy
    # and applies as fitness.
    def teamFitness(self, programOutput):
        for i in range(len(teams)):
            output = teams[i].calcOutput(programOutput, actionList)
            correctness = output == labels
            teams[i].fitness = len(correctness[correctness == True])

# Object representing a team.
class Team:
    
    def __init__(self, numLearners):
        self.learners = np.zeros(numLearners)
        fitness = 0
       
    # Given the output of the programs, compute what the team does for each exemplar.
    # ProgramOutput is program x exemplars.
    # outcomes is 1D, with # elements = # exemplars.
    def calcOutput(self, programOutput, actionList):
        winningBids = np.argmax(programOutput.T * self.learners)
        outcomes = actionList[winningBids]
        return outcomes
        
# Relates what programs suggest what actions.
# Here as a class for sake of organization, though not absolutely necessary.
class ActionList:

    # Will need more creative means to assign this.
    def __init__(self, action):
        self.actions = actions