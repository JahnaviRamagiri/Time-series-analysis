import numpy as np
import warnings

warnings.filterwarnings("ignore")


def cal_avg_method(train_set, test_set):
    x = np.array(train_set)
    train_pred = [np.mean(x[:t]) for t in range(len(x))]
    test_pred = len(test_set) * [np.mean(x)]

    return train_pred, test_pred


def cal_naive_method(train_set, test_set):
    train_pred = [float('nan')]
    train_pred[1:] = train_set[:-1]
    test_pred = len(test_set) * [train_set[-1]]

    return train_pred, test_pred


def cal_drift_method(train_set, test_set):
    train_pred = [float('nan'), float('nan')]
    train_pred[2:] = [train_set[i-1] + (train_set[i-1] - train_set[0])/(i - 1) for i in range(2, len(train_set))]
    yT = train_set[-1]
    test_pred = [yT + h*(yT - train_set[0])/(len(train_set) - 1) for h in range(len(test_set))]

    return train_pred, test_pred


def cal_ses(train_set, test_set, alpha = 0.5):
    train_pred = []
    for t in range(len(train_set)):
        if t == 0:
            train_pred.append(train_set[0])

        else:
            train_pred.append(alpha * train_set[t-1] + (1 - alpha) * train_pred[t-1])

    test_pred = len(test_set) * [alpha * train_set[-1] + (1 - alpha) * train_pred[-1]]

    return train_pred, test_pred