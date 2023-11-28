import os
import numpy as np
import math
import warnings
from scipy import signal
from modules import arma, statistics as st

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=FutureWarning)


def print_title(title, pattern = "*", pattern_length = 20, num_blank_lines = 1):
    """

    :param title: String to be printed
    :param pattern: Pattern preceeding and Succeding the String in the title
    :param pattern_length: Length of pattern
    :param num_blank_lines: Total blank lines before and after the Title
    :return: Null
    """
    print((num_blank_lines//2 + 1) * "\n", pattern_length * pattern, title, pattern_length * pattern, num_blank_lines//2 * "\n")


def get_file_path(file_name):
    """

    :param file_name:
    :return:
    """
    dir_path = os.getcwd()
    print("Current work Directory", dir_path)
    file_path = dir_path + os.sep + file_name
    print("File Path is ", file_path)

    return file_path


def get_log_transform(df_col):
    return np.log(df_col)


def check_stationarity(df_col, title, thres = 0.05, order=0):
    st.cal_rolling_mean_var(df_col[order:], title)
    st.ADF_Cal(df_col[order:], title, thres)
    st.kpss_test(df_col[order:], title, thres)


def get_mean(x):
    return sum(x)/len(x)


def gen_random_variable(mean = 0, variance = 1, observations = 1000):
    return np.random.normal(mean, variance, observations)


# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
    means[i] = sum(col_values) / float(len(dataset))
    return means


# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
    stdevs[i] = sum(variance)
    stdevs = [math.sqrt(x / (float(len(dataset) - 1))) for x in stdevs]
    return stdevs


def inv_std(data, mean, std):
    return data * std + mean


# def cal_stat_significance(ry)

def get_df_info(df, title):
    print_title(title, pattern="-", pattern_length=10)
    print("The Data Frame Contains Columns", list(df.columns))
    print("Size of the Data Frame", df.shape)
    print(df.head())


def print_observation(text):
    print("OBSERVATION: ", text)


def print_2f(text):
    print(f"{text:.2f}")


def print_3f(text):
    print(f"{text:.3f}")


def get_sse(theta, y, na):
    e = get_etheta(theta, y, na)
    return e, e.T @ e


def pad_np_arrays(array1, array2):
    len1, len2 = len(array1), len(array2)

    if len1 < len2:
        array1 = np.pad(array1, (0, len2 - len1), mode='constant')
    elif len2 < len1:
        array2 = np.pad(array2, (0, len1 - len2), mode='constant')

    return array1, array2

def get_etheta(theta, y, na):

    an = theta[:na]
    bn = theta[na:]
    arparams = np.array(an)
    maparams = np.array(bn)

    den = np.r_[1, arparams]
    num = np.r_[1, maparams]
    den, num = pad_np_arrays(den, num)

    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    e = np.ndarray.flatten(e)

    return e

def get_deltheta(A, g, mu, n):
    I = np.identity(n)
    del_theta = np.linalg.inv(A + mu * I) @ g
    return del_theta


def get_x(theta, delta, y, na, nb):
    # i =  5
    # e_0 = e_theta - np.concatenate(([e_theta[0] + delta], e_theta[1:]))
    # x_0 = np.array(e_0) / delta
    n = na + nb
    e_theta = get_etheta(theta, y, na)
    x = []
    for i in range(n):
        theta_new = theta.copy()
        theta_new[i] = theta[i] + delta
        e_td = get_etheta(theta_new, y, na)
        x.append(np.stack((e_theta - e_td)/delta, axis=0))
    # x = [(np.array(e_theta - np.concatenate((e_theta[:i], [e_theta[i] + delta], e_theta[i+1:])))/delta).T for i in range(0, len(e_theta))]
    X = np.array(x)
    X = X.T
    return X

def cal_var(y, theta, na, mean_WN, var_WN, data_samples):

    # utils.print_title("Contruct generated ARMa Process")
    est_an = theta[:na]
    est_bn = theta[na:]
    est_arma_process = arma.get_process(est_an, est_bn)
    mean_y = mean_WN * (1 + np.sum(est_bn)) / (1 + np.sum(est_an))
    est_y = est_arma_process.generate_sample(data_samples, scale=np.sqrt(var_WN)) + mean_y

    error = y - est_y
    err_var = np.var(error)
    # print("Variance of Error is ", err_var)

    return err_var

def get_LM(na, nb, y, N, mean_WN, var_WN, delta = 1e-06, max_iter = 100, thres = 1e-03, mu_max = 10000):
    print(f"LM Parameters: na {na}, nb {nb}, delta {delta}, max_iter {max_iter}, thres {thres},  mu_max {mu_max} ")
    sse_list = []
    n = na + nb
    mu = 0.1

    # step 0
    theta = np.zeros(na + nb)

    # step 1
    e_theta, sse_theta = get_sse(theta, y, na)
    X = get_x(theta, delta, y, na, nb)
    A = X.T @ X
    g = X.T @ e_theta
    sse_list.append(sse_theta)

    # step 2
    d_theta = get_deltheta(A, g, mu, n)
    theta_new = theta + d_theta
    _, sse_new = get_sse(theta_new, y, na)
    sse_list.append(sse_new)

    iter_ = 0
    theta_p = 0
    sd = 0
    cov_theta = 0
    if iter_ < max_iter:
        if sse_new < sse_theta:
            mag_del = np.linalg.norm(d_theta)
            if mag_del < thres:
                theta_p = theta_new
                sd = sse_new/(N-n)
                cov_theta = sd * sd * np.linalg.inv(A)
                return theta_p, sd, cov_theta

            else:
                theta = theta_new
                mu = mu/10

        while sse_new >= sse_theta:
            mu = mu*10
            if mu > mu_max:
                print("Error mu > mu_max")
                theta_p = theta_new
                sd = sse_new / (N - n)
                cov_theta = sd * sd * np.linalg.inv(A)
                return theta_p, sd, cov_theta

            d_theta = get_deltheta(A, g, mu, n)
            theta_new = theta + d_theta
            _, sse_new = get_sse(theta_new, y, na)

        # iter_var = cal_var(y, theta_new, na, mean_WN, var_WN, N)
        sse_list.append(sse_new)
        # print("Var", iter_var)
        # print("Next Iteration", iter_)
        iter_ += 1
        print("Iteration", iter_)
        if iter_ > max_iter:
            print("Error iter_ > max_iter")
            theta_p = theta_new
            sd = sse_new / (N - n)
            cov_theta = sd * sd * np.linalg.inv(A)
            return theta_p, sd, cov_theta

        theta = theta_new
        # Step 1
        e_theta, sse_theta = get_sse(theta, y, na)
        X = get_x(theta, delta, y, na, nb)
        A = X.T @ X
        g = X.T @ e_theta

        # Step 2
        d_theta = get_deltheta(A, g, mu, n)
        theta_new = theta + d_theta
        _, sse_new = get_sse(theta_new, y, na)

    theta_p = theta_new
    sd = sse_new / (N - n)
    cov_theta = sd * sd * np.linalg.inv(A)
    return sse_list, theta_p, sd, cov_theta

def get_CI(theta, cov_matrix):
    ci_min = theta - 2*np.sqrt(np.linalg.norm(cov_matrix))
    ci_max = theta + 2*np.sqrt(np.linalg.norm(cov_matrix))

    return ci_min, ci_max







