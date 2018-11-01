import numpy as np
from utils import prepare_data, softmax, y_to_matrix
import argparse
from sklearn.metrics import classification_report


def mini_batch(batch_size, X, y):
    indices = np.arange(0, len(X))
    np.random.shuffle(indices)
    indices = indices[:batch_size]
    X_shuffled = [X[i] for i in indices]
    y_shuffled = [y[i] for i in indices]

    return np.array(X_shuffled), np.array(y_shuffled)


def loss(X, y, weights):
    p = softmax(weights, X)
    summa = -np.sum(y * np.log(p))
    return summa / len(X)


def gradient(X, weights, y):
    p = softmax(weights, X)
    return np.dot(np.transpose(p - y), X) / (len(X))

def grad_step(X, y, current_weights, step):
    while True:
        current_loss = loss(X, y, current_weights)
        grad = gradient(X, current_weights, y)
        new_weights = current_weights - step * grad
        new_loss = loss(X, y, new_weights)
        if new_loss >= current_loss:
            step = step / 2
        else:
            difference = current_loss - new_loss
            return new_weights, new_loss, difference


def train(X_train, y_train, num_iter, step):
    np.random.seed(0)
    w = np.random.normal(scale=0.001, size=(10, X_train.shape[1]))
    eps = 1e-7
    for i in range(num_iter):
        w, loss, difference = grad_step(X_train, y_train, w, step)
        if i % 10 == 0:
            print(i, 'loss:', loss)
        if difference <= eps:
            break
    return w



def main():
    path_to_train_x = 'dataset/train-images.idx3-ubyte'
    path_to_train_y = 'dataset/train-labels.idx1-ubyte'
    path_to_save = 'result'

    parser = argparse.ArgumentParser(description='mnist train')
    parser.add_argument('--x_train_dir=', dest='x_train_dir', default=path_to_train_x, type=str)
    parser.add_argument('--y_train_dir=', dest='y_train_dir', default=path_to_train_y, type=str)
    parser.add_argument( '--model_output_dir=',dest='model_output_dir', default=path_to_save, type=str)
    parser.add_argument('--mini_batch_num=', dest='mini_batch_num', type=int, default=1000, help='По дефолту 1000 эпох')
    args = parser.parse_args()

    X, y = prepare_data(args.x_train_dir, args.y_train_dir)
    weights = train(X, y, args.mini_batch_num, 0.5)
    np.save(args.model_output_dir, weights)

    p = softmax(weights, X)
    y_pred = np.argmax(p, axis=1)
    y_pred_matrix = y_to_matrix(y_pred, 10)
    print(classification_report(y, y_pred_matrix))

if __name__ == '__main__':
    main()
