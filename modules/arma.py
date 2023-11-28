import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
from modules import plot
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def get_moving_average(m, data, fm = 2):
    data_length = len(data)
    if m % 2 == 0:
        k = m // 2
        ma = [round(1/2 * 1/m * (sum([sum(data[t-k+i : t+k+i])
                                      for i in range(fm)])),2)
              for t in range(k, data_length-k)]
        ma = [float('nan')] * k + ma + [float('nan')] * k
        # Case Even
        print(ma)
    else:
        # Case Odd
        k = (m - 1) // 2
        # ma = [0] * data_length
        ma = [round(1/m * sum(data[t-k:t+k+1]),2) for t in range(k, data_length-k)]
        ma = [float('nan')] * k + ma + [float('nan')] * k
        print(ma)

    return ma


def cal_AR(a, method,  T = 100, mean = 0, var = 1):
    """

    :param a: AR coefficients starting from y(t-1)
    :param T: No. Of Samples
    :param mean: Mean of White Noise
    :param var: Variance of White Noise
    :return: Time Series Y

    Prints mean and variance of the time series
    """
    e = np.random.normal(mean, var, T)
    y = np.zeros(len(e))
    an = len(a)

    # Hardcoding it to y(t) - 0.5y(t-1) - 0.2y(t-2) = e(t)

    for t in range(len(e)):
        if t== 0:
            y[t]=e[t]

        elif t == 1:
            y[t] = e[t] + 0.5 * y[t-1]

        else:
            y[t] = e[t] + 0.5 * y[t-1] + 0.2 * y[t-2]

    print(f"Equation for {T, mean, var}")
    print(f'Sampled mean is {np.mean(y)}')
    print(f'Sampled var is {np.var(y)}')
    plt.plot(y)
    plt.show()

    return y


def get_arma(num, den, T, mean, var, title = ""):

    e = np.random.normal(mean, var, T)
    sys = (num, den, 1)
    _, y = signal.dlsim(sys, e)
    # print('for loop simulation', y[:5])
    y = np.ndarray.flatten(y)

    date_range = pd.date_range(start="01-01-2000", periods = len(y), freq='D')
    df = pd.DataFrame(y, index = date_range)

    print(f"Equation for {T, mean, var}")
    print(f'Sampled mean is {np.mean(y)}')
    print(f'Sampled var is {np.var(y)}')

    plot.plot_variable_graph(df, 0, "Time", "Y", title, sample_size=100)

    return y

# def get_moving_average(m, data, fm = 2):
#     data_length = len(data)
#     if m % 2 == 0:
#         k = m // 2
#         ma = [round(1/2 * (1/m * sum(data[t-k : t+k]) +
#                            1/m * sum(data[t-k+1 : t+k+1])), 2)
#               for t in range(k,data_length-k)]
#         ma = [float('nan')] * k + ma + [float('nan')] * k
#         # Case Even
#         print(ma)
#     else:
#         # Case Odd
#         k = (m - 1) // 2
#         # ma = [0] * data_length
#         ma = [round(1/m * sum(data[t-k:t+k+1]),2) for t in range(k, data_length-k)]
#         ma = [float('nan')] * k + ma + [float('nan')] * k
#         print(ma)
#
#     return ma

def get_process(an,bn):

    arparams = np.array(an)
    maparams = np.array(bn)

    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]

    arma_process = sm.tsa.ArmaProcess(ar, ma)

    return arma_process



