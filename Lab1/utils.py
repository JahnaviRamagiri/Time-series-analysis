import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import math
from scipy import signal
import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=FutureWarning)


def print_title(title, pattern = "*", pattern_length = 20, num_blank_lines = 1):
    print((num_blank_lines//2 + 1 )* "\n", pattern_length * pattern, title, pattern_length * pattern, num_blank_lines//2 * "\n")


def get_file_path(file_name):
    dir_path = os.getcwd()
    print("Current work Directory", dir_path)
    file_path = dir_path + os.sep + file_name
    print("File Path is ", file_path)

    return file_path


def plot_data(df):
    sales = df["Sales"]
    adbudget = df["AdBudget"]
    gdp = df["GDP"]

    df["Date"] = pd.to_datetime(df["Date"], format = "%m/%d/%Y")

    # Plot Data
    plt.figure(figsize = (10,6))
    plt.plot(df["Date"], sales, label = "Sales")
    plt.plot(df["Date"], adbudget, label = "AdBudget")
    plt.plot(df["Date"], gdp, label="GDP")
    plt.xlabel("Date")
    plt.ylabel("USD($)")
    plt.grid(True)
    plt.title("Sales, AdBudget, GDP vs Time")
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.show()


def plot_graph(df, col_list, x_label, x_interval, y_label, title, sample_size):
    x_col = df.index

    # Plot Data
    plt.figure(figsize = (10,6))

    for col in col_list:
        plt.plot(x_col[:sample_size], df[col][:sample_size], label=col)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.title(title)
    plt.legend()

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%B %Y"))
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=x_interval))

    plt.show()


def get_statistics(df, col_list):
    mean = np.mean(df)
    var = np.var(df)
    std = np.std(df)
    median = df.median()

    for ind, col in enumerate(col_list[1:]):
        print(f"The {col} mean is: {mean[ind]:.2f} and the variance is {var[ind]:.2f} with standard deviation {std[ind+1]:.2f} median:{median[ind]:.2f}")


def get_rolling_mean(df_col):
    acc_sum = 0
    roll_mean = []
    for ind, i in enumerate(df_col):
        acc_sum += i
        roll_mean.append(acc_sum/(ind+1))
    return roll_mean


def get_rolling_var(df_col):
    roll_var = []
    for ind, i in enumerate(df_col):
        df_var = df_col.head(ind).var()
        roll_var.append(df_var)

    return roll_var


def cal_rolling_mean_var(df_col, title):
    r_mean = get_rolling_mean(df_col)
    r_var = get_rolling_var(df_col)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rolling mean
    axes[0].plot(range(len(r_mean)), r_mean, label='Rolling Mean')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'Rolling Mean {title}')

    # Plot rolling variance
    axes[1].plot(range(len(r_var)), r_var, label='Rolling Variance')
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title(f'Rolling Variance {title}')
    if "Log" not in title:
        axes[1].set_ylim(0, max(r_var[2:]) + 1)

    # plt.tight_layout()
    plt.show()

    return r_mean, r_var


def ADF_Cal(x, title):
    print(f"ADF-test for {title}")
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))



def kpss_test(timeseries, title):
    print(f'KPSS Test {title} ')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print("\n")
    print (kpss_output)


def get_differencing(df_col):
    diff_list = []
    # last_ind

    for ind, ele in enumerate(df_col):
        if ind == 0:
            diff_list.append(float('nan'))
            continue
        diff_list.append(df_col[ind] - df_col[ind-1])
    return diff_list


def get_log_transform(df_col):
    return np.log(df_col)


def check_stationarity(df_col, title, thres = 0.05, order=0):
    cal_rolling_mean_var(df_col[order:], title)
    ADF_Cal(df_col[order:], title, thres)
    kpss_test(df_col[order:], title, thres)


def get_mean(x):
    return sum(x)/len(x)


def correlation_coefficent_cal(x, y):

    if len(x) != len(y):
        raise ValueError("Both datasets must have the same length")

    # Convert x and y to numpy arrays for efficient calculations
    x = np.array(x)
    y = np.array(y)

    # Calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    r_num = np.sum((x - mean_x) * (y - mean_y))
    r_den = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))

    return r_num/r_den


def gen_random_variable(mean = 0, variance = 1, observations = 1000):

    return np.random.normal(mean, variance, observations)


def auto_corr_func(data, max_lag, title = "ACF", plot = True):
    n = len(data)
    lags = np.arange(-max_lag, max_lag + 1)
    acf_values = []

    # Calculate the mean of the data
    mean = np.mean(data)

    # Calculate the autocorrelation for each lag in the list
    for lag in lags:
        if lag <= 0:
            # Negative lag
            numerator = np.sum((data[:n + lag] - mean) * (data[-lag:] - mean))
        else:
            # Positive lag
            numerator = np.sum((data[:-lag] - mean) * (data[lag:] - mean))

        denominator = np.sum((data - mean) ** 2)

        acf = numerator / denominator
        acf_values.append(acf)

    if plot:
        (markers, stemlines, baseline) = plt.stem(lags, acf_values, markerfmt='o')
        plt.setp(markers, color = 'red', marker = 'o')
        plt.setp(baseline, color='grey', linewidth=2, linestyle='-')
        m = 1.96/np.sqrt(n)
        # print("m", m)
        plt.axhspan(-m, m,alpha = 0.2, color = 'blue')
        plt.xlabel("Lags")
        plt.ylabel("Magnitude")
        # plt.xticks(lags)
        plt.title(title)
        # plt.grid()
        plt.show()

    return acf_values


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


def plot_forecast(train_set, test_set, test_forecast, title):
    plt.plot(train_set, marker='o', color='blue', label='Training Set')

    # Plot the test set
    test_x = np.arange(len(train_set), len(train_set) + len(test_set))
    plt.plot(test_x, test_set, marker='s', color='green', label='Test Set')

    # Plot the h-step forecast
    plt.plot(test_x, test_forecast, marker='^', color='red', label='Forecast (h-step)')

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.legend()
    plt.title(title)
    plt.grid()

    # Show the plot
    plt.show()

def get_error_df(data_points,  data_pred):
    df = pd.DataFrame()
    df["Data"] = data_points
    df["Prediction"] = data_pred
    df["Error"] = df["Data"] - df["Prediction"]
    df["SSE"] = df["Error"] ** 2

    return df


def cal_var(x):
    return np.var(x)


def cal_q_value(x, lags):
    acf = auto_corr_func(x, lags, plot = False)
    q = lags * sum(np.square(acf[lags+1:]))

    return q


def cal_metrics(train_set, test_set, train_pred, test_forecast, title = ""):
    # train_set = np.array(train_set)

    train_df = get_error_df(train_set,  train_pred)
    pred_mse = sum(train_df['SSE'][2:])/len(train_df['SSE']-2)
    pred_var = cal_var(train_df["Error"][2:])
    pred_mean = sum(train_df['Error'][2:])/len(train_df['SSE']-2)
    q_pred = cal_q_value(train_df["Error"][2:], 5)

    test_df = get_error_df(test_set, test_forecast)
    forecast_mse = sum(test_df['SSE'])/len(test_df['SSE'])
    forecast_var = cal_var(test_df["Error"])

    auto_corr_func(train_df["Error"], 10, f"Residual Error ACF: {title}", plot=True)

    print(train_df)
    print(f"MSE of Prediction Errors: {pred_mse:.2f}")
    print(f"Mean of Residuals: {pred_mean:.2f}")
    print(f"Variance of Prediction Error: {pred_var:.2f}")
    print(f"Q Value: {q_pred:.2f}")

    print(test_df)
    print(f"MSE of Forecast Errors: {forecast_mse:.2f}")
    print(f"Variance of Forecast Error: {pred_var:.2f}")

    return pred_mse, forecast_mse, pred_var, forecast_var, q_pred


def plot_ses(train_set, test_set, alphas):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, alpha in enumerate(alphas):
        ax = axes[i // 2, i % 2]  # Select the appropriate subplot
        train_pred, test_forecast = cal_ses(train_set, test_set, alpha)

        # Plot the training set
        ax.plot(range(len(train_set)), train_set, marker='o', color='blue', label='Training Set')

        # Plot the test set
        ax.plot(range(len(train_set), len(train_set) + len(test_set)), test_set, marker='s', color='green',
                label='Test Set')

        # Plot the h-step forecast
        ax.plot(range(len(train_set), len(train_set) + len(test_set)), test_forecast, marker='^', color='red',
                label='Forecast (h-step)')

        # Add title, labels, and legend
        ax.set_title(f'Alpha = {alpha}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


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


# standardize dataset
def standardize_dataset(dataset):
    means = column_means(dataset)
    stdevs = column_stdevs(dataset, means)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


def get_OLS(x, y, selected_features):
    # Fit a model with all selected features
    X = x[selected_features].values
    X = sm.add_constant(X)  # Add an intercept
    y = y.values
    model = sm.OLS(y, X).fit()

    # Calculate AIC, BIC, and Adjusted R-squared
    aic = model.aic
    bic = model.bic
    adj_r2 = model.rsquared_adj
    p_values = model.pvalues[1:]

    print(f"AIC : {aic:.2f}")
    print(f"BIC : {bic:.2f}")
    print(f"Adjusted R2 : {adj_r2:.2f}")

    return model, p_values


def get_back_step_regression(x, y, features):
    # Initialize the best model criteria
    best_aic = float("inf")
    best_bic = float("inf")
    best_adj_r2 = -float("inf")
    best_model = None

    while len(features) > 1:
        model, p_values = get_OLS(x, y, features)

        # Calculate AIC, BIC, and Adjusted R-squared
        aic = model.aic
        bic = model.bic
        adj_r2 = model.rsquared_adj

        # Check if the current model has better criteria
        if aic < best_aic or bic < best_bic or adj_r2 > best_adj_r2:
            best_aic = aic
            best_bic = bic
            best_adj_r2 = adj_r2
            best_model = model

        else:
            break

        print(f"Dropping \"{features[p_values.argmax()]}\" with highest p-value {p_values.max():.2f}")
        print("\n")
        features.remove(features[p_values.argmax()])

    print("\nFinal Selected features:", features)
    print(f"Best AIC {best_aic:.2f}")
    print(f"Best BIC: {best_bic:.2f}")
    print(f"Best Adjusted R-squared: {best_adj_r2:.2f}")

    return best_model, features


def get_vif_reduction(x, y, features):
    # Initialize the best model criteria
    best_aic = float("inf")
    best_bic = float("inf")
    best_adj_r2 = -float("inf")
    best_model = None

    while len(features) > 1:
        model, p_values = get_OLS(x, y, features)

        # Calculate AIC, BIC, and Adjusted R-squared
        aic = model.aic
        bic = model.bic
        adj_r2 = model.rsquared_adj

        # Check if the current model has better criteria
        if aic < best_aic or bic < best_bic or adj_r2 > best_adj_r2:
            best_aic = aic
            best_bic = bic
            best_adj_r2 = adj_r2
            best_model = model

        else:
            break

        vif = pd.Series([variance_inflation_factor(x[features].values, i) for i in range(x[features].shape[1])], index=x[features].columns)
        # Identify the feature with the highest VIF
        feature_to_remove = vif.idxmax()

        # Remove the feature with the highest VIF
        features.remove(feature_to_remove)
        print(f"Dropping \"{feature_to_remove}\" with highest vif-value {vif.max()}")

    print("\nFinal Selected features:", features)
    print(f"Best AIC {best_aic:.2f}")
    print(f"Best BIC: {best_bic:.2f}")
    print(f"Best Adjusted R-squared: {best_adj_r2:.2f}")

    return best_model, features


def inv_std(data, mean, std):
    return data * std + mean


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

def plot_ma(ma_list, data, sample_size = 50, fm = 2):

    output_df = pd.DataFrame()
    output_df["Data"] = data

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, m in enumerate(ma_list):

        ax = axes[i // 2, i % 2]  # Select the appropriate subplot
        ma = get_moving_average(m, data, fm)
        # output_df[f"{m}"] = ma
        title = (f"{fm}X{m} MA" if m % 2 == 0 else f"{m}")
        output_df[title] = ma

        # Plot the training set
        ax.plot(range(sample_size), data[:sample_size], label='Original Data')

        # Plot the test set
        ax.plot(range(sample_size), ma[:sample_size], label=f'Moving Average {title}')

        ax.plot(range(sample_size), data[:sample_size] - ma[:sample_size], label='Detrended Data')

        # Add title, labels, and legend
        ax.set_title(f'Moving Average {title}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return output_df

def cal_strength(T, S, R):
    Ft = np.maximum(0, 1 - np.var(R)/(np.var(T+R)))
    Fs = np.maximum(0, 1 - np.var(R)/np.var(S+R))

    print(f"The Strength of Trend for this Dataset is {Ft:.2f}")
    print(f"The Strength of Seasonality for this dataset is {Fs:.2f}")

    return Ft, Fs


def cal_partial_correlation(rxy, ryz, rxz):
    rnum = rxy - rxz * ryz
    rden = (np.sqrt(1- rxz ** 2)) * np.sqrt(1-ryz ** 2)

    return rnum/rden

# def cal_stat_significance(ry)

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

    plot_variable_graph(df, 0, "Time", "Y", title, sample_size=100)
    # plt.plot(df)
    # plt.show()

    return y


def get_df_info(df, title):
    print_title(title, pattern="-", pattern_length=10)
    print("The Data Frame Contains Columns", list(df.columns))
    print("Size of the Data Frame", df.shape)
    print(df.head())


def print_observation(text):
    print("OBSERVATION: ", text)


def plot_variable_graph(df, target, x_label, y_label, title, sample_size = 100):
    # Plot Data
    plt.figure(figsize = (10,6))
    plt.plot(df.index[:sample_size], df[target][:sample_size], label=target)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()


def get_mean_var(y_list):
    return [np.mean(y) for y in y_list], [np.var(y) for y in y_list]

