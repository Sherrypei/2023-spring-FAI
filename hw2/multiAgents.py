# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autographing was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# reference : https://github.com/jiminsun/berkeley-cs188-pacman/blob/master/hw2/multiagent/multiAgents.py
import random

import util
from game import Agent
from game import Directions
from pacman import GameState

MAXN = 999999.0


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return MAXN
        newGhostDistance = [util.manhattanDistance(newPos, newGhost.getPosition()) for newGhost in newGhostStates]
        newFoodDistance = [util.manhattanDistance(newPos, foods) for foods in newFood.asList()]
        score = successorGameState.getScore() + min(newGhostDistance)
        # print(score,minGhostDistance,minFoodDistance)
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            score += 100
        if newPos in currentGameState.getCapsules():
            score += 200
        score = score + 100 / min(newFoodDistance)
        if action == Directions.STOP:
            score -= 30

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            return self.evaluationFunction(gameState), "Stop"
        utility = util.Counter()
        for action in gameState.getLegalActions(0):
            utility[action] = self.minvalue(gameState.generateSuccessor(0, action), 1, 0)
        return utility.argMax()

    def maxvalue(self, state, agent, depth):
        value = -MAXN
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        else:
            for action in state.getLegalActions(agent):
                value = max(value, self.minvalue(state.generateSuccessor(agent, action), agent + 1, depth))
            return value

    def minvalue(self, state, agent, depth):
        value = MAXN
        if state.getLegalActions(agent):
            if agent == state.getNumAgents() - 1:
                for action in state.getLegalActions(agent):
                    value = min(self.maxvalue(state.generateSuccessor(agent, action), 0, depth + 1), value)
            else:
                for action in state.getLegalActions(agent):
                    value = min(value, self.minvalue(state.generateSuccessor(agent, action), agent + 1, depth))
            return value
        return self.evaluationFunction(state)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #        util.raiseNotDefined()
        utility = util.Counter()
        a = -MAXN
        b = MAXN
        for action in gameState.getLegalActions(0):
            value = self.minvalue(gameState.generateSuccessor(0, action), 1, 0, a, b)
            utility[action] = value
            a = max(a, value)
        return utility.argMax()

    def maxvalue(self, state, agent, depth, a, b):
        value = -MAXN
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        else:
            for action in state.getLegalActions(agent):
                value = max(value, self.minvalue(state.generateSuccessor(agent, action), agent + 1, depth, a, b))
                if value > b:
                    return value
                a = max(a, value)
            return value

    def minvalue(self, state, agent, depth, a, b):
        value = MAXN
        if state.getLegalActions(agent):
            if agent == state.getNumAgents() - 1:
                for action in state.getLegalActions(agent):
                    value = min(value, self.maxvalue(state.generateSuccessor(agent, action), 0, depth + 1, a, b))
                    if value < a:
                        return value
                    b = min(b, value)
            else:
                for action in state.getLegalActions(agent):
                    value = min(value, self.minvalue(state.generateSuccessor(agent, action), agent + 1, depth, a, b))
                    if value < a:
                        return value
                    b = min(b, value)
            return value
        return self.evaluationFunction(state)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        utility = util.Counter()
        for action in gameState.getLegalActions(0):
            utility[action] = self.expvalue(gameState.generateSuccessor(0, action), 1, 0)
        return utility.argMax()

    def expvalue(self, state, agent, depth):
        value = 0.0
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        else:
            prob = 1.0 / len(state.getLegalActions(agent))
            if agent == state.getNumAgents() - 1:
                for action in state.getLegalActions(agent):
                    value += prob * self.maxvalue(state.generateSuccessor(agent, action), 0, depth + 1)
            else:
                for action in state.getLegalActions(agent):
                    value += prob * self.expvalue(state.generateSuccessor(agent, action), agent + 1,
                                                  depth)
            return value

    def maxvalue(self, state, agent, depth) -> float:
        value = -MAXN
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        else:
            for action in state.getLegalActions(agent):
                value = max(value, self.expvalue(state.generateSuccessor(agent, action), agent + 1, depth))
            return value


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here, so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def count_walls_between(pos, food):
        x, y = pos
        foodx, foody = food
        foodx, foody = int(foodx), int(foody)

        return sum([wx in range(min(x, foodx), max(x, foodx) + 1) and
                    wy in range(min(y, foody), max(y, foody) + 1) for (wx, wy) in wall])

    #   util.raiseNotDefined()
    if currentGameState.isWin():
        return MAXN
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    FoodDistance = [util.manhattanDistance(Pos, food) for food in Food]
    Capsule = currentGameState.getCapsules()
    Score = float(currentGameState.getScore())
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    wall = currentGameState.getWalls().asList()
    closestFood = sorted(FoodDistance)
    closeFoodDistance = sum(closestFood[-5:])
    closestFoodDistance = sum(closestFood[-3:])
    ghostDistance = [
        util.manhattanDistance(Pos, ghost.getPosition()) + 2 * count_walls_between(Pos, ghost.getPosition())
        for ghost in GhostStates]
    Score += 0.5 * ScaredTimes[0] + 2.0 / len(Food) - len(Capsule) + min(min(ghostDistance),
                                                                         6) + 2.0 / closeFoodDistance + 2.5 / closestFoodDistance
    return Score


# Abbreviation
better = betterEvaluationFunction
