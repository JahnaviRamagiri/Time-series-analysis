import os
import numpy as np
import math
import warnings
import statistics as st

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
