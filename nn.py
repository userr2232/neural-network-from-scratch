import numpy as np
from numpy.random import randn, permutation

class NN(object):
    alpha = 0.5;
    epochs = 30;
    def __init__(self, *dims):
        assert(len(dims) > 1), "You should specify the dimension of the input and of at least one layer."
        self.L = len(dims);
        self.dims = dims;
        self.ws = [ randn(dim_i, dim_j) for dim_i, dim_j in zip(dims[:-1], dims[1:]) ];
        self.bs = [ randn(dim, 1) for dim in dims[1:] ];
        self.best_ws = self.ws;
        self.best_bs = self.bs;
        self.loss = np.inf;
        self.score = 0;
    
    def load(self, X, target):
        perm = permutation(len(X));
        self.X = X[perm];
        self.target = target[perm];

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x));

    @staticmethod
    def sigmoid_prime(x):
        g = NN.sigmoid(x)
        return g * (1 - g);

    def predict(self, x):
        for w, b in zip(self.ws, self.bs):
            x = self.sigmoid(np.dot(x, w) + b.transpose());
        return x;

    def l2_loss(self, X, target):
        dim = self.target.shape[1];
        N = len(X);
        loss = 0;
        for x, r in zip(X, target):
            loss += np.sum(np.square(self.predict(x) - r));
        loss /= 2 * N;
        return loss;

    def confusion_matrix(self, X, target):
        dim = self.target.shape[1];
        matrix= np.zeros((dim, dim));
        for x, r in zip(X, target):
            j = list(r).index(1);
            pred = np.rint(self.predict(x)[0]);
            pred = list(pred);
            i = pred.index(max(pred))
            matrix[i][j] += 1;
        return matrix;

    def tp(self, conf_matrix):
        dim = conf_matrix.shape[0];
        results = [];
        for i in range(dim):
            results.append(conf_matrix[i][i]);
        return results;

    def tn(self, conf_matrix):
        dim = conf_matrix.shape[0];
        results = [];
        for i in range(dim):
            matrix_w_o_feature = np.delete(conf_matrix, i, 0);
            matrix_w_o_feature = np.delete(conf_matrix, i, 1);
            results.append(np.sum(matrix_w_o_feature));
        return results;

    def fp(self, conf_matrix):
        dim = conf_matrix.shape[0];
        results = [];
        for i in range(dim):
            matrix_w_o_column = np.delete(conf_matrix, i, 1);
            results.append(np.sum(matrix_w_o_column[i]));
        return results;
    
    def fn(self, conf_matrix):
        dim = conf_matrix.shape[0];
        fns = 0;
        results = [];
        for i in range(dim):
            matrix_w_o_row = np.delete(conf_matrix, i, 0);
            matrix_w_o_row = matrix_w_o_row.transpose();
            results.append(np.sum(matrix_w_o_row[i]));
        return results;

    def tp_fp_tn_fn(self, X, target):
        conf_matrix = self.confusion_matrix(X, target);
        return (self.tp(conf_matrix),
                self.fp(conf_matrix),
                self.tn(conf_matrix),
                self.fn(conf_matrix));
    
    def accuracy(self, tp, fp, tn, fn):
        return (tp + tn) / ( tp + fp + tn + fn );

    def precision(self, tp, fp, tn, fn):
        return 0 if tp + fp == 0 else tp / ( tp + fp );

    def recall(self, tp, fp, tn, fn):
        return 0 if tp + fn == 0 else tp / ( tp + fn );

    def f1_score(self, tp, fp, tn, fn):
        p = self.precision(tp, fp, tn, fn);
        r = self.recall(tp, fp, tn, fn);
        return 0 if p + r == 0 else 2 * p * r / ( p + r );

    def scores(self, X, target):
        tps, fps, tns, fns = self.tp_fp_tn_fn(X, target);
        dim = len(tps);
        precisions = [];
        recalls = [];
        f1_scores = [];
        tp_sum = 0;
        fp_sum = 0;
        tn_sum = 0;
        fn_sum = 0;
        for tp, fp, tn, fn in zip(tps, fps, tns, fns):
            tp_sum += tp;
            fp_sum += fp;
            tn_sum += tn;
            fn_sum += fn;
            precisions.append(self.precision(tp, fp, tn, fn));
            recalls.append(self.precision(tp, fp, tn, fn));
            f1_scores.append(self.f1_score(tp, fp, tn, fn));
        
        return { 'accuracy': self.accuracy(tp_sum, fp_sum, tn_sum, fn_sum),
                    'precision': np.sum(precisions) / dim,
                    'recall': np.sum(recalls) / dim,
                    'f1_score': np.sum(f1_scores) / dim };

    # Backpropagation algorithm
    # Inspired by the pseudocode presented on fig. 18.24 
    # on Artificial Intelligence. A Modern Approach by Russell, S. Norvig, P. (2010)
    # However, this algorithm has an important modification:
    # It can use all the dataset or many random subsets generated on each epoch (SGD)
    # batch_size is used to indicate if it should use all data or only small datasets
    def back_prop_learning(self, criterion, batch_size = None):
        print("Training net with layers {}".format(self.dims));
        X = self.X;
        target = self.target;
        N = len(X);
        for epoch in range(self.epochs):
            if batch_size and batch_size < N:
                    perm = permutation(N);
                    X = self.X[perm][:batch_size];
                    N = len(X);
                    target = self.target[perm][:batch_size];
            for x, r in zip(X, target):
                a = x;
                as_ = [ x ];
                ins = [ x ];
                for w, b in zip(self.ws, self.bs):
                    in_ = np.dot(a, w) + b.transpose();
                    ins.append(in_);
                    a = self.sigmoid(in_);
                    as_.append(a);
                y = as_[-1];
                d = self.sigmoid_prime(ins[-1]) * (r - y);
                ds = [];
                ds.append((d[0][:,np.newaxis], d));
                for l in reversed(range(self.L-1)):
                    b_delta = np.empty(self.dims[l+1]);
                    w_delta = np.empty(self.dims[l]);
                    for i in range(self.dims[l]):
                        w_delta[i] = self.sigmoid_prime(ins[l][0][i]) * np.sum(ds[0][1] * self.ws[l][i]);
                    for j in range(self.dims[l+1]):
                        b_delta[j] = self.sigmoid_prime(1) * np.sum(ds[0][0] * self.bs[l][j]);
                    b_delta = np.array(b_delta[:,np.newaxis]);
                    ds.insert(0, (np.array(b_delta), np.array([w_delta])));
                for i in range(len(self.ws)):
                    self.ws[i] = np.add(self.ws[i], self.alpha * np.dot(as_[i].transpose(), ds[i+1][1]));
                    self.bs[i] = np.add(self.bs[i], self.alpha * ds[i][0]);
            if (epoch + 1) % 5 == 0:
                print("Epoch {}/{}".format(epoch + 1, self.epochs));
            if criterion:
                scores = self.scores(X, target);
                score = scores[criterion];
                if score > self.score + 0.1:
                    self.best_ws = self.ws;
                    self.best_bs = self.bs;
                    d = (score - self.score) * 0.05
                    if self.alpha - d > 0:
                        self.alpha -= d;
                elif score < self.score - 0.1:
                    print("Premature training exit on epoch {}".format(epoch + 1));
                    self.ws = self.best_ws;
                    self.bs = self.best_bs;
                    break;
                self.score = score;
                if (epoch + 1) % 5 == 0:
                    print("{}: {}".format(criterion, score));
            else:
                loss = self.l2_loss(X, target);
                if loss <= self.loss:
                    self.best_ws = self.ws;
                    self.best_bs = self.bs;
                    d = (self.loss - loss) * 0.5
                    if self.alpha - d > 0:
                        self.alpha -= d;
                elif loss > self.loss + 0.1:
                    print("Premature training exit on epoch {}".format(epoch + 1));
                    self.ws = self.best_ws;
                    self.bs = self.best_bs;
                    break;
                self.loss = loss;
                if (epoch + 1) % 5 == 0:
                    print("loss:", loss)