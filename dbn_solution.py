"""
Deep Belief Networks using Deep Belief Network
and TensorFlow. The DBN folder contains the deep-belief-network package
originally downloaded from https://github.com/albertbup/deep-belief-network
on March 29, 2017, and modified slightly to support python3. The
DBN folder is covered under the MIT license.
"""
from __future__ import print_function
from __future__ import division

from dbn.tensorflow import SupervisedDBNClassification
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

train_ex = 8000


dbn = SupervisedDBNClassification(
    hidden_layers_structure=[1024, 1024],
    learning_rate_rbm=0.1,
    learning_rate=0.1,
    n_epochs_rbm=10,
    n_iter_backprop=100,
    batch_size=100,
    activation_function='sigmoid',
    dropout_p=0.2
)


def unpickle(file):
    """Load data"""
    import pickle
    with open(file, 'rb') as source:
        ret_dict = pickle.load(source, encoding='bytes')
    return ret_dict


def get_data():
    """
    Loads the data in.
    """
    tmp = unpickle("CIFAR-3.pickle")
    labels = []
    for index in range(len(tmp['y'])):
        if tmp['y'][index, 0] == 1:
            #airplane
            labels.append(1)
        elif tmp['y'][index, 1] == 1:
            #dog
            labels.append(2)
        else:
            #boat
            labels.append(3)

    x_train = tmp['x'][:train_ex]
    x_train /= 255
    y_train = labels[:train_ex]
    x_test = tmp['x'][train_ex:]
    x_test /= 255
    y_test = labels[train_ex:]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data()

dbn.fit(x_train, y_train)

predictions = dbn.predict(x_test)
accuracy = accuracy_score(y_test, list(predictions))
print('Accuracy: {0}'.format(accuracy))
