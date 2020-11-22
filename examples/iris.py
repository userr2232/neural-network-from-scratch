import sys
sys.path.insert(1, '../')
from nnsearch import NN_complete_search

from sklearn.preprocessing import OneHotEncoder, normalize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris();
X = iris.data;
X = normalize(X);
X = X[:, np.newaxis];
Y = iris.target[:, np.newaxis];
encoder = OneHotEncoder(sparse=False);
Y = encoder.fit_transform(Y);
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25);
print("\n" + "+" * 40 + "Iris dataset" + "+" * 40)
print("Starting training...")

results = NN_complete_search(X_train, Y_train, X_train.shape[-1], Y_train.shape[-1]);
print("Completed training.")
for result in results:
    criterion, nn = result;
    if criterion == 'loss':
        print("Minimum L2 loss among all Neural Networks trained: {}".format(nn.loss));
        print("Layers dimensions: {}".format(nn.dims))
        print("prediction for {} with real class {}: {}".format(X_test[0], Y_test[0], nn.predict(X_test[0])));
        print("L2 Loss of best artificial neural network with test data:", nn.l2_loss(X_test, Y_test));
    else:
        print("Maximum {} among all Neural Networks trained: {}".format(criterion, nn.score));
        print("Layers dimensions: {}".format(nn.dims))
        print("prediction for {} with real class {}: {}".format(X_test[0], Y_test[0], nn.predict(X_test[0])));
        print("{} of best artificial neural network with test data:".format(criterion), nn.scores(X_test, Y_test)[criterion]);