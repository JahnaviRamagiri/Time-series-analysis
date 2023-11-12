import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

import simple_forecasting_methods as sfm
import arma

warnings.filterwarnings("ignore")


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
    """
    :param df: DataFrame
    :param col_list: List of Features to be plotted
    :param x_label: Label of x-axis
    :param x_interval: Interval for x-axis
    :param y_label: Label for y-axis
    :param title: Plot Title
    :param sample_size: Number of samples to be plotted
    :return: null

    PLots all the features of the dataset. The index of the dataset is taken as X column.
    """
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


def plot_rolling_mean_var(r_mean, r_var, title):
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


def plot_ses(train_set, test_set, alphas):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, alpha in enumerate(alphas):
        ax = axes[i // 2, i % 2]  # Select the appropriate subplot
        train_pred, test_forecast = sfm.cal_ses(train_set, test_set, alpha)

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


def plot_ma(ma_list, data, sample_size = 50, fm = 2):

    output_df = pd.DataFrame()
    output_df["Data"] = data

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, m in enumerate(ma_list):

        ax = axes[i // 2, i % 2]  # Select the appropriate subplot
        ma = arma.get_moving_average(m, data, fm)
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
