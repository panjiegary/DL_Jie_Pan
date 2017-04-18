"""
Convolution neural network.
"""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 3
epochs = 100

train_ex = 8000
img_h, img_w = 32, 32


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
    x_train = tmp['x'][:train_ex]
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_train /= 255
    y_train = tmp['y'][:train_ex]

    x_test = tmp['x'][train_ex:]
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    x_test /= 255
    y_test = tmp['y'][train_ex:]

    return x_train, y_train, x_test, y_test


def convolution():
    """
    Keras follows the layers principle, where each layer
    is independent and can be stacked and merged together.
    The Sequential model assumes that there is one long
    stack, with no branching.
    """
    x_train, y_train, x_test, y_test = get_data()

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_h, img_w, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Stochastic Gradient Descent
    sgd = SGD(lr=0.01, momentum=0.9)
    es = EarlyStopping(monitor='val_loss',
                       patience=5,  # epochs to wait after min loss
                       min_delta=0.0001)  # anything less than this counts as no change

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[es])

    score = model.evaluate(x_test, y_test)
    print('Test loss: {0}'.format(score[0]))
    print('Test accuracy: {0}'.format(score[1]))
    model.save_weights('model.hdf5')

    plt.figure('Predictions', facecolor='gray')
    plt.set_cmap('gray')

    predictions = model.predict(x_test, verbose=0)
    for i in range(5):
        subplt = plt.subplot(int(i / 5) + 1, 5, i + 1)
        # no sense in showing labels if they don't match the letter
        if predictions[i, 0] >= predictions[i, 1]:
            index = 0
            title = 'airplane'
        else:
            index = 1
            title = 'dog'
        if predictions[i, index] < predictions[i, 2]:
            title = 'boat'
        subplt.set_title('Prediction: {0}'.format(title))
        subplt.axis('off')
        letter = x_test[i]
        subplt.matshow(np.reshape(letter, [img_h, img_w]))
        plt.draw()


if __name__ == '__main__':
    convolution()
    plt.show()

