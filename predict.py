import numpy as np
import argparse
from utils import prepare_data, model, y_to_matrix
from sklearn.metrics import classification_report

def main():
    path_to_x = 'dataset/t10k-images.idx3-ubyte'
    path_to_y = 'dataset/t10k-labels.idx1-ubyte'
    path_to_model = 'result.npy'

    parser = argparse.ArgumentParser(description='predict.py')
    parser.add_argument('--x_test_dir=', dest='x_test_dir', default=path_to_x, type=str)
    parser.add_argument('--y_test_dir=',dest='y_test_dir',default=path_to_y, type=str)
    parser.add_argument('--model_input_dir=', dest='model_input_dir',default=path_to_model, type=str)
    args = parser.parse_args()

    weights = np.load(args.model_input_dir)

    X, y = prepare_data(args.x_test_dir, args.y_test_dir)
    y_pred = model(X, weights)
    y_pred_matrix = y_to_matrix(y_pred, 10)
    print(classification_report(y, y_pred_matrix))



if __name__ == '__main__':
    main()
