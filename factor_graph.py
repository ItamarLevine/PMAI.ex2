###############################################################################
# factor graph data structure implementation 
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

import functools
import numpy as np
from factors import *


class FactorGraph:
    def __init__(self, numVar=0, numFactor=0):
        '''
        var list: index/names of variables

        domain list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]

        varToFactor: list of lists, it has the same length as the number of variables. 
                     varToFactor[i] is a list of the indices of Factors that are connected to variable i

        factorToVar: list of lists, it has the same length as the number of factors. 
                     factorToVar[i] is a list of the indices of Variables that are connected to factor i

        factors: a list of Factors

        messagesVarToFactor: a dictionary to store the messages from variables to factors,
                            keys are (src, dst), values are the corresponding messages of type Factor

        messagesFactorToVar: a dictionary to store the messages from factors to variables,
                            keys are (src, dst), values are the corresponding messages of type Factor
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [[0,1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}
        self.belief = [None for _ in range(numFactor)]
        self.belief_vars = [None for _ in range(numVar)]

    def addFactor(self, factor):
        '''
        :param factor: a Factor object
        '''
        self.factors.append(Factor(factor))
        assert len(self.factors) <= len(self.factorToVar)
        for var_idx in factor.scope:
            self.varToFactor[var_idx].append(len(self.factors) - 1)
        self.factorToVar[len(self.factors) - 1] = factor.scope

    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigment
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val[tuple(a[f.scope])]
        return output
    
    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        param - type: type of messages. "varToFactor" is the messages from variables to factors; 
                    "factorToVar" is the message from factors to variables
        return: message from src to dst
        
        In this function, the message will be initialized as an all-one vector (normalized) if 
        it is not computed and used before. 
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        if type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]

    def runParallelLoopyBP(self, iterations):
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. The only values 
        you should update in this function are self.messagesVarToFactor and self.messagesFactorToVar. 
        
        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.

        Note: You can also calculate the marginal MAPs after each iteration here...
        '''
        ###############################################################################
        # initialize messages = 1
        for i, f in enumerate(self.factors):
            for var in f.scope:
                self.getInMessage(i, var, "factorToVar")
                self.getInMessage(var, i)
        for it in range(iterations):
            # print('.', end='', flush=True)
            if (it+1) % 5 == 0:
                print(it+1, end='', flush=True)
            ##############################################################
            for var in self.var:
                for factor in self.varToFactor[var]:
                    # create variable-to-factor messages
                    other_factors = np.setdiff1d(self.varToFactor[var],factor)
                    m = self.messagesFactorToVar[(other_factors[0], var)]
                    for i in range(1, len(other_factors)):
                        m = m.multiply(self.messagesFactorToVar[(other_factors[i], var)])
                    m = m.normalize()
                    self.messagesVarToFactor[(var, factor)] = m
                    # create factor-to-variable messages
                    other_vars = np.setdiff1d(self.factorToVar[factor], var)
                    m = self.factors[factor]
                    for i in range(0, len(other_vars)):
                        m = m.multiply(self.messagesVarToFactor[(other_vars[i], factor)])
                    m = m.marginalize_all_but([var])
                    m = m.normalize()
                    self.messagesFactorToVar[(factor, var)] = m
            ##############################################################
        # calculate_beliefs
        for i in range(len(self.var)):
            m = self.messagesFactorToVar[(self.varToFactor[i][0],self.var[i])]
            for j in range(1,len(self.varToFactor[i])):
                m = m.multiply(self.messagesFactorToVar[(self.varToFactor[i][j], self.var[i])])
            m = m.normalize()
            self.belief_vars[i] = m
        #print()

        ###############################################################################

    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probability of a single variable after running
        loopy belief propagation. (This method assumes runParallelLoopyBP has been run)

        param - var: a single variable index
        return: numpy array of size 2 containing the marginal probabilities 
                that the variable takes the values 0 and 1
        
        example: 
        factor_graph.estimateMarginalProbability(0) =
        [0.2, 0.8]
        '''
        if type(var) != int:
            var = var[0]
        return [self.belief_vars[var].val[i] for i in [0,1]]

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP assignment for each variable.
        You may utilize the method estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        factor_graph.getMarginalMAP() =
        [0, 1, 0, 0]
        '''
        ###############################################################################
        return [np.argmax(self.estimateMarginalProbability(self.var[i])) for i in range(len(self.var))]
        ###############################################################################

    def print(self):
        print('Variables:')
        for i in range(len(self.var)):
            print('  Variable {}: {}'.format(i, self.var[i]))
            print('     In factors:', self.varToFactor[i])
        print('Factors:')
        for i, f in enumerate(self.factors):
            print('  Factor {}: {}'.format(i, f))
            print('     vars=', self.factorToVar[i])
            print('     scope=', f.scope)
            print('     card=', f.card)
            print('     val=', f.val)
