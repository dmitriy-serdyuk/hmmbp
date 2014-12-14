import argparse
import numpy as np
import scipy.io


def read_data(filename):
    data = scipy.io.loadmat(filename)
    data = np.array(data['price_move'])[:, 0]
    n = len(data)
    return n, data


def bprop(n, data, q_value=None, prob_x1=None, prob_x_x=None,
          prob_x_y=None):

    # x = 'bad' <=> 0, x = 'good' <=> 1
    # y = -1 <=> 0, y = 1 <=> 1
    message_y_x = np.zeros((n, 2))
    message_x_fort = np.zeros((n, 2))
    message_x_back = np.zeros((n, 2))
    message_x_y = np.zeros((n, 2))

    # Probability x_{t + 1} given x_t
    if prob_x_x is None:
        prob_x_x = np.array([[0.8, 0.2],
                             [0.2, 0.8]])
    # Probability x_t given y_t
    if prob_x_y is None:
        prob_x_y = np.array([[q_value, (1 - q_value)],
                             [(1 - q_value), q_value]])
    # Probability x_1
    prob_x1 = np.array([1 - prob_x1, prob_x1])

    # Backward pass
    for i in reversed(xrange(0, n)):
        if data[i] == -1:
            message_y_x[i, :] = prob_x_y[:, 0]
        else:
            message_y_x[i, :] = prob_x_y[:, 1]

        if i == n - 1:
            message_x_back[n - 1, :] = (message_y_x[n - 1, 0] * prob_x_x[0, :] +
                                        message_y_x[n - 1, 1] * prob_x_x[1, :])
            continue

        message_x_back[i, :] = (message_y_x[i, 0] * message_x_back[i + 1, 0] * prob_x_x[0, :] +
                                message_y_x[i, 1] * message_x_back[i + 1, 1] * prob_x_x[1, :])


    # Forward pass
    for i in xrange(0, n):
        if i == 0:
            message_x_y[0, :] = (prob_x_y[1, :] * message_x_back[1, 1] * prob_x1[1] +
                                 prob_x_y[0, :] * message_x_back[1, 0] * prob_x1[0])
            message_x_fort[0, :] = (prob_x_x[:, 1] * message_y_x[0, 1] * prob_x1[1] +
                                    prob_x_x[:, 0] * message_y_x[0, 0] * prob_x1[0])
            continue
        if i == n - 1:
            message_x_y[n - 1, :] = (prob_x_y[1, :] * message_x_fort[n - 2, 1] +
                                     prob_x_y[0, :] * message_x_fort[n - 2, 0])
            message_x_fort[n - 1, :] = (prob_x_x[:, 1] * message_y_x[n - 1, 1] * message_x_fort[n - 2, 1] +
                                        prob_x_x[:, 0] * message_y_x[n - 1, 0] * message_x_fort[n - 2, 0])
            continue
        message_x_y[i, :] = (prob_x_y[1, :] * message_x_back[i + 1, 1] * message_x_fort[i - 1, 1] +
                             prob_x_y[0, :] * message_x_back[i + 1, 0] * message_x_fort[i - 1, 0])
        message_x_fort[i, :] = (prob_x_x[:, 1] * message_y_x[i, 1] * message_x_fort[i - 1, 1] +
                                prob_x_x[:, 0] * message_y_x[i, 0] * message_x_fort[i - 1, 0])

    # Normalisation
    message_y_x /= np.sum(message_y_x, axis=1).reshape((-1, 1))
    message_x_fort /= np.sum(message_x_fort, axis=1).reshape((-1, 1))
    message_x_back /= np.sum(message_x_back, axis=1).reshape((-1, 1))
    message_x_y /= np.sum(message_x_y, axis=1).reshape((-1, 1))
    return message_y_x, message_x_fort, message_x_back, message_x_y


def compute_llh(data, params, prob_x1=None):
    if prob_x1 is None:
        prob_x1 = np.array([1 - 0.8, 0.8])
    n = data.shape[0]
    a, b, c, d = params
    prob_x_x = np.array([[a, (1 - b)],
                         [(1 - a), b]])
    prob_x_y = np.array([[c, (1 - d)],
                         [(1 - c), d]])
    messages = bprop(n, data, prob_x_x=prob_x_x, prob_x_y=prob_x_y,
                     prob_x1=prob_x1)

    x_good = compute_prob(n, *messages)
    size = data.shape[0]

    sum = 0.
    for xi in [0, 1]:
        sum += np.log(prob_x1[xi]) * x_good[0, xi]

    for i in xrange(1, size):
        for xi in [0, 1]:
            for xi1 in [0, 1]:
                if np.any(x_good[i] == 0):
                    print i, x_good[i]
                if prob_x_x[xi, xi1] < 1e-10:
                    return 0
                #    prob_x_x[xi, xi1] = 1e-10
                sum += (np.log(prob_x_x[xi, xi1]) * prob_x_x[xi, xi1] *
                        x_good[i - 1, xi1] / x_good[i, xi])

    for i in xrange(size):
        for xi in [0, 1]:
            sum += (np.log(prob_x_y[xi, (data[i] + 1) / 2]) *
                    x_good[i, xi])

    return sum / n


def compute_prob(n, message_y_x, message_x_fort, message_x_back, message_x_y):
    x_good = np.zeros((n, 2))
    x_good[0, :] = message_y_x[0, :] * message_x_back[1, :]
    x_good[n - 1, :] = message_y_x[n - 1, :] * message_x_fort[n - 1, :]
    for i in xrange(1, n - 1):
        x_good[i, :] = message_x_fort[i - 1, :] * message_x_back[i + 1, :] * message_y_x[i, :]

    # Normalisation
    x_good += 1e-10
    x_good /= np.sum(x_good, axis=1).reshape((-1, 1))
    return x_good


def recompute_params(n, data, messages, params):
    x_good = compute_prob(n, *messages)
    a, b, c, d = params
    sum = np.zeros(2)
    for i in xrange(1, n):
        sum += x_good[i - 1, :] / x_good[i, 1]

    new_a = sum[1] * a / (sum[1] * a + sum[0] * (1 - a))

    sum2 = np.zeros(2)
    for i in xrange(1, n):
        sum2 += x_good[i - 1, :] / x_good[i, 0]

    new_b = sum2[0] * b / (sum2[0] * b + sum2[1] * (1 - b))

    sum3 = np.zeros(2)
    for i in xrange(0, n):
        if data[i] == 1:
            sum3[1] += x_good[i, 1]
        if data[i] == -1:
            sum3[0] += x_good[i, 0]

    sum4 = np.sum(x_good, axis=0)

    new_c = sum3[1] / sum4[1]

    new_d = sum3[0] / sum4[0]

    return [new_a, new_b, new_c, new_d]


def infer(data):
    n = data.shape[0]
    a = 0.8
    b = 0.8
    c = 0.9
    d = 0.9
    params = [a, b, c, d]
    prob_x1 = 0.8
    prob_x1_m = np.array([1 - prob_x1, prob_x1])
    while True:
        a, b, c, d = params
        prob_x_x = np.array([[a, (1 - b)],
                             [(1 - a), b]])
        prob_x_y = np.array([[c, (1 - d)],
                             [(1 - c), d]])
        messages = bprop(n, data, prob_x_x=prob_x_x, prob_x_y=prob_x_y,
                         prob_x1=prob_x1)
        params = recompute_params(n, data, messages, params)
        x_good = compute_prob(n, *messages)
        yield params, x_good


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
    all_params = []
    all_llh = []
    all_x = []
    n, data = read_data('sp500.mat')
    data_train = data[:30]
    data_valid = data[30:]

    for i, (params, x_good) in enumerate(infer(data)):
        #print i
        all_params += [params]
        train_llh = compute_llh(data, params)
        valid_llh = compute_llh(data_valid, params)

        #print 'llh', i, llh
        all_llh += [[train_llh, valid_llh]]
        all_x += [x_good]
        if i > 50:
            break
    print all_params
    print all_llh
    print all_x
    #args = parse_args()
    #bprop(args.input, args.q_value, args.prob_x1)