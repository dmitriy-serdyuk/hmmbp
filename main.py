import argparse
import numpy as np
import scipy.io


def read_data(filename):
    data = scipy.io.loadmat(filename)
    data = np.array(data['price_move'])[:, 0]
    n = len(data)
    return n, data


def main(filename, q_value, prob_x1):
    n, data = read_data(filename)
    message_y_x = np.zeros(n)
    message_x_fort = np.zeros(n)
    message_x_back = np.zeros(n)
    message_x_y = np.zeros(n)

    if data[n - 1] == -1:
        message_y_x[n - 1] = 1 - q_value
    else:
        message_y_x[n - 1] = q_value
    message_x_back[n - 1] = message_y_x[n - 1] * 0.8 + (1 - message_y_x[n - 1]) * 0.2
    for i in reversed(xrange(0, n - 1)):
        if data[i] == -1:
            message_y_x[i] = 1 - q_value
        else:
            message_y_x[i] = q_value
        message_x_back[i] = (message_y_x[i] * message_x_back[i + 1] * 0.8 +
                             (1 - message_y_x[i]) * (1 - message_x_back[i + 1])
                             * 0.2)

    message_x_y[0] = ((1 - q_value) * (1 - message_x_back[0]) * (1 - prob_x1) +
                      q_value * message_x_back[0] * prob_x1)
    message_x_fort[0] = 0.2 * (1 - message_y_x[0]) + 0.8 * message_y_x[0]
    for i in xrange(1, n):
        message_x_y[i] = ((1 - q_value) * (1 - message_x_back[i]) *
                          (1 - message_x_fort[i - 1]) +
                          q_value * message_x_back[i] * message_x_fort[i - 1])
        message_x_fort[i] = (0.2 * (1 - message_y_x[i]) *
                             (1 - message_x_fort[i - 1]) +
                             0.8 * message_y_x[i] * message_x_fort[i - 1])

    return message_y_x, message_x_fort, message_x_back, message_x_y


def compute_prob(n, message_y_x, message_x_fort, message_x_back):
    x_good = np.zeros(n)
    x_good[0] = message_y_x[0] * message_x_back[0]
    x_good[n - 1] = message_y_x[n - 1] * message_x_fort[n - 1]
    for i in xrange(1, n - 1):
        x_good[i] = message_x_fort[i - 1] * message_x_back[i] * message_y_x[i]
    return x_good


def parse_args():
    parser = argparse.ArgumentParser(
        "Run sum-product algorithm")
    parser.add_argument("--input",
                        type=str, help="Input .mat file", default='file.mat')
    parser.add_argument("--q-value",
                        type=float, default=0.7,
                        help="Value of q")
    parser.add_argument("--prob-x1",
                        type=float, default=0.2,
                        help="Probability of x1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.q_value, args.prob_x1)