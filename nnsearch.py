from nn import NN;

from itertools import product
import numpy as np


# This algorithm searches for the best neural network
# X, Y: the dataset
# input_dim, output_dim: are the dimensions of the input and output layers respectively
# stochatic: whether to use SGD or not
# batch_size: batch size (only use when stochastic=True)
# max_dim: max dimension of any hidden layer
# max_hidden_layers: max number of hidden layers of a nn in the search space
def NN_complete_search(X, Y, input_dim, output_dim, stochastic = False, batch_size = 500, max_dim = 5, max_hidden_layers = 2):
    criteria = ['accuracy', 'precision', 'recall', 'f1_score', None ];
    results = [];
    for criterion in criteria:
        min_loss = np.inf;
        max_score = 0;
        best_nn = None;
        for hidden_layers in range(max_hidden_layers + 1):
            dims = [ np.arange(1, max_dim + 1) ] * hidden_layers
            products = product(*dims);
            for p in products:
                net = NN(input_dim, *p, output_dim);
                net.load(X, Y);
                net.back_prop_learning(criterion, batch_size if stochastic else None);
                if criterion:
                    nn_score = net.score;
                    nn_dims = net.dims;
                    if nn_score > max_score:
                        max_score = nn_score;
                        best_nn = net;
                else:
                    nn_loss = nn.loss;
                    nn_dims = nn.dims;
                    if nn_loss < min_loss:
                        min_loss = nn_loss;
                        best_nn = net;
        results.append((criterion if criterion else 'loss', best_nn));
    return results;