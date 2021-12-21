# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            bestUtility = util.Counter()
            for state in self.mdp.getStates():
                optimalAction = self.getAction(state)
                if not self.mdp.isTerminal(state):
                    bestUtility[state] = self.getQValue(state, optimalAction)
            self.values = bestUtility





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # each state have a probability to be accomplished and we obtail q value as the equation
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #TODO: returns the Q-value of the (state, action) pair given by the value function given by self.values
        TransitionStatesProbs=self.mdp.getTransitionStatesAndProbs(state,action)
        qvalue=0
        for nextstate, prob in TransitionStatesProbs:
            qvalue+=((self.values[nextstate]* self.discount)+self.mdp.getReward(state,action,nextstate))*prob
        return qvalue
        #return self.getQValue(state,action)
        #util.raiseNotDefined()

# `   best action
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #we should return the best action in a given state
        #TODO: computes the best action according to the value function given by self.values.
        if self.mdp.isTerminal(state):
            return None
        utilities=util.Counter()
        actions=self.mdp.getPossibleActions(state)
        if len(actions)==0:
            return
        for action in actions:
            #print(self.getQValue(state,action))
            utilities[action]+=self.getQValue(state,action)
        return utilities.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)
    # retruns the best Action
    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    # the best value
    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    #we will update one state in each iteration
    def runValueIteration(self):
        iter=0
        while (iter<self.iterations):
            for state in self.mdp.getStates():
                bestUtility = util.Counter()
                #optimalAction = self.getAction(state)
                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        bestUtility[action] = self.computeQValueFromValues(state, action)
                    self.values[state] = bestUtility[bestUtility.argMax()]

                iter+=1
                if (iter>=self.iterations):
                    return



        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):


        all_states_predecessors=dict()
        #bestvalues=util.Counter()
        queue=util.PriorityQueue()
        states = self.mdp.getStates()

        for state in states:
            all_states_predecessors[state] = set()

        for s in states:
            QValues=util.Counter()
            for action in self.mdp.getPossibleActions(s):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                    if prob != 0:
                        all_states_predecessors[nextState].add(s)
                QValues[action] = self.computeQValueFromValues(s, action)

            if not self.mdp.isTerminal(s):
                maxQvalue=QValues[QValues.argMax()]
                diff=abs(self.values[s]-maxQvalue)
                queue.update(s,-diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                return
            s=queue.pop()
            if not self.mdp.isTerminal(s):
                bestValue = util.Counter()
                for a in self.mdp.getPossibleActions(s):
                    bestValue[a] = self.computeQValueFromValues(s, a)
                self.values[s] = bestValue[bestValue.argMax()]


            s_predeseccors=all_states_predecessors[s]
            for pre in s_predeseccors:
                bestUtility = util.Counter()
                # optimalAction = self.getAction(state)
                if not self.mdp.isTerminal(pre):
                    for action in self.mdp.getPossibleActions(pre):
                        bestUtility[action] = self.computeQValueFromValues(pre, action)
                bestQvalue=bestUtility[bestUtility.argMax()]
                diff=abs(self.values[pre]-bestQvalue)
                if diff>self.theta:
                    queue.update(pre,-diff)
        "*** YOUR CODE HERE ***"

