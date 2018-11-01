import numpy as np
from sklearn.metrics import classification_report
import struct
import gzip

def parse_file(path, descriptor):
    with descriptor(path, 'rb') as f:
        size = struct.unpack('>xxxB', f.read(4))[0]

        shape = struct.unpack(f'>{"I"*size}', f.read(4 * size))

        shape = (shape[0], shape[1] * shape[2]) if len(shape) == 3 else (shape[0],)
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    return data

def read_mnist(path):
    if path.endswith('.gz'):
        descriptor = gzip.open
    else:
        descriptor = open

    return parse_file(path, descriptor)


def softmax(weights, X):
    d = np.exp(np.dot(X, np.transpose(weights)))
    summ = np.sum(d, axis=1)
    for i in range(len(weights)):
        d[:, i] = d[:, i] / summ
    return d

def model(X, weights):
    p = softmax(weights, X)
    return np.argmax(p, axis=1)

def one_hot(y, number_of_classes=10):
    y_one_hot = np.eye(number_of_classes)[y[:]]
    return y_one_hot

def accuracy(predicted_values, true_values):
    predicted_answers = np.argmax(predicted_values, axis=1)
    true_answers = np.argmax(true_values, axis=1)
    correct_results = 0
    for i, val in enumerate(predicted_answers):
        correct_results += predicted_answers[i] == true_answers[i]
    return correct_results / len(true_answers)


def prepare_data(features_path, labels_path):
    X = read_mnist(features_path)
    X = X.reshape((-1, 784)).astype(float) / 256.0
    bias_column = np.ones((X.shape[0], 1))
    X = np.hstack((bias_column, X))

    y = read_mnist(labels_path)
    y = one_hot(y)

    return X, y


def y_to_matrix(y, number_of_classes):
    y1 = np.zeros((len(y), number_of_classes))
    for ind, elem in enumerate(y):
        y1[ind][elem] = 1.
    return y1

