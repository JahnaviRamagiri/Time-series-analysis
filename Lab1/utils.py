import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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




