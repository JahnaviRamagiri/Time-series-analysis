import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=FutureWarning)


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


def get_statistics(df):
    mean = df.mean()
    var = df.var()
    std = df.std()
    median = df.median()
    print(f"The Sales mean is: {mean.Sales:.2f} and the variance is {var.Sales:.2f} with standard deviation {std.Sales:.2f} median:{median.Sales:.2f}")
    print(f"The AdBudget mean is: {mean.AdBudget:.2f} and the variance is {var.AdBudget:.2f} with standard deviation {std.AdBudget:.2f} median:{median.AdBudget:.2f}")
    print(f"The GDP mean is: {mean.GDP:.2f} and the variance is {var.GDP:.2f} with standard deviation {std.GDP:.2f} median:{median.GDP:.2f}")


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
    axes[1].set_ylim(0, max(r_var[2:]) + 5)

    # plt.tight_layout()
    plt.show()

    return r_mean, r_var


def ADF_Cal(x, title, thres):
    print(f"ADF-test for {title}")
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[1] <= 1 - thres:
        print(f"{title} is stationary \n")
    else:
        print(f"{title} is not stationary \n")


def kpss_test(timeseries, title, thres):
    print('Results of KPSS Test: ')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print("\n")
    print (kpss_output)

    if kpss_output[1] < 1-thres:
        print(f"{title} is stationary\n")
    else:
        print(f"{title} is not stationary \n")







