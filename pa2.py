###############################################################################
# HUJI 67800 PMAI 2021 - Programming Assignment 2
# Original authors: Ya Le, Billy Jun, Xiaocheng Li, Yiftach Beer
###############################################################################
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices
    return values:
        G: generator matrix
        H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
        yTilde_i is obtained by flipping y_i with probability epsilon
    '''
    ###############################################################################
    yTilde = y ^ np.random.choice([0, 1], p=[1-epsilon, epsilon], size=y.shape)
    ###############################################################################
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    :param - yTilde: observed codeword
        type: numpy.ndarray containing 0's and 1's
        shape: 2N
    :param - H parity check matrix
             type: numpy.ndarray
             shape: N x 2N
    :param epsilon - the probability that each bit is flipped to its complement
    return G FactorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors
    '''

    N, M = H.shape
    graph = FactorGraph(numVar=M, numFactor=N+M)
    graph.var = range(M)

    ##############################################################
    # Add unary factors
    for i in range(M):
        if yTilde[i] == 0:
            val = [1 - epsilon, epsilon]
        else:
            val = [epsilon, 1 - epsilon]
        graph.addFactor(Factor(scope=[i],card=[2], val=np.array(val),name=f"unary{i}"))
    # Add parity factors
    for i in range(N):
        scope = np.where(H[i,:] == 1)[0]
        all_placments = list(itertools.product([0,1], repeat=len(scope)))
        val = np.zeros([2]*len(scope))
        for placement in all_placments:
            val[placement] = 1 - (np.sum(np.array(placement)) % 2)
        graph.addFactor(Factor(scope=scope, card=[2]*len(scope), val=val, name=f"parity{i}"))
    ##############################################################
    return graph


def q1():
    yTilde = np.array([[1, 1, 1, 1, 1, 1]]).reshape(6,1)
    H = np.array([[0, 1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    graph = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    # Report their weights (joint probability values) respectively.
    ytest1 = [0,1,0,0,0,0]
    ytest2 = [1,1,1,1,1,1]
    ytest3 = [0,1,1,1,0,1]
    ##############################################################
    print(
        graph.evaluateWeight(ytest1),
        graph.evaluateWeight(ytest2),
        graph.evaluateWeight(ytest3))
    ##############################################################


def q3(epsilon):
    '''
    In q(3), we provide you an all-zero initialization of message x, you should
    apply noise on y to get yTilde, and then do loopy BP to obtain the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    ##############################################################
    y_tilde = applyChannelNoise(y, epsilon)
    graph = constructFactorGraph(y_tilde, H, epsilon)
    graph.runParallelLoopyBP(50)
    marginals = np.apply_along_axis(graph.estimateMarginalProbability, 0, np.arange(len(y)).reshape(1,len(y)))

    ##############################################################
    plt.figure()
    plt.title(f'q(3): Marginals for all-ones input, epsilon={epsilon}')
    plt.plot(range(len(y)), marginals[1,:], '.-')
    plt.savefig(f'q3_{epsilon}.png', bbox_inches='tight')
    plt.show()
    ##############################################################


def q4(numTrials, epsilon, iterations=50):
    '''
    param - numTrials: how many trials we repeat the experiments
    param - epsilon: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    plt.figure()
    plt.title(f'q(4): Hamming distances, epsilon={epsilon}')
    for trial in range(numTrials):
        print('Trial number', trial)
        ##############################################################
        # apply noise, construct the graph
        # run loopy while retrieving the marginal MAP after each iteration
        # calculate Hamming distances and plot
        y_tilde = applyChannelNoise(y, epsilon)
        graph = constructFactorGraph(y_tilde, H, epsilon)
        hamming_distances = []
        for _ in range(iterations):
            graph.runParallelLoopyBP(1)
            estimate_y = graph.getMarginalMAP()
            hamming_distances.append(np.sum(estimate_y != y.reshape(len(y),)))
        plt.plot(range(iterations),hamming_distances)
    plt.grid(True)
    plt.savefig(f'q4_{epsilon}.png', bbox_inches='tight')
    plt.show()
    ##############################################################


def q6(epsilon):
    '''
    param - epsilon: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = np.load('image.npy')

    N = G.shape[1]
    x = img.reshape(N, 1)
    y = encodeMessage(x, G)
    y_tilde = applyChannelNoise(y, epsilon)
    graph = constructFactorGraph(y_tilde, H, epsilon)
    plt.figure()
    plt.title(f'q(6): Image reconstruction, epsilon={epsilon}')
    show_image(y_tilde, 0, 'Input')

    plot_iters = [0, 1, 3, 5, 10, 20, 30]

    ##############################################################
    # Todo: your code starts here
    graph.runParallelLoopyBP(0)
    result = graph.getMarginalMAP()
    show_image(np.array(result), 1, f'Iter {plot_iters[0]}')
    for i in range(1,31):
        graph.runParallelLoopyBP(1)
        if i in plot_iters:
            result = graph.getMarginalMAP()
            show_image(np.array(result), plot_iters.index(i)+1, f'Iter {i}')
    ##############################################################
    plt.savefig(f'q6_{epsilon}.png', bbox_inches='tight')
    plt.show()
    ################################################################


def show_image(output, loc, title, num_locs=8):
    image = output.flatten()[:len(output)//2]
    image_radius = int(np.sqrt(image.shape))
    image = image.reshape((image_radius, image_radius))
    ax = plt.subplot(1, num_locs, loc + 1)
    ax.set_title(title)
    ax.imshow(image)
    ax.axis('off')


if __name__ == "__main__":
    print('Running q(1): Should see 0.0, 0.0, >0.0')
    q1()

    print('Running q(3):')
    for epsilon in [0.05, 0.06, 0.08, 0.1]:
        q3(epsilon)

    print('Running q(4):')
    for epsilon in [0.05, 0.06, 0.08, 0.1]:
        q4(10, epsilon)

    print('Running q(6):')
    q6(0.06)

    print('All done.')
