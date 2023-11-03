import pandas as pd
import math
import numpy as np
import utils
from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt

if __name__ == '__main__':

    file_path = utils.get_file_path("daily-min-temperatures.csv")

    utils.print_title("Load Data")
    df = pd.read_csv(file_path)
    Temp = pd.Series(np.array(df["Temp"]), index=pd.date_range('1981-01-01'
                                                         , periods = len(df["Temp"]),
                                                         freq ='d'),
                     name='Temp')
    df = pd.DataFrame(Temp)
    print(df.head())
    df.plot()

    utils.print_title("Moving average")
    order = int(input("Input the order of Moving Average"))
    fold_order = 2
    if order == 1 or order == 2:
        print("Order 1 and 2 are not accepted")
    elif order % 2 == 0:
        fold_order = int(input("Input folding order"))

    ma = utils.get_moving_average(order, df["Temp"], fold_order)
    utils.print_title("Moving average - ODD ")
    ma_list = [3, 5, 7, 9]
    odd_df = utils.plot_ma(ma_list, df["Temp"], 50)
    print(odd_df)

    utils.print_title("Moving average - EVEN ")
    ma_list = [4, 6, 8, 10]
    even_df = utils.plot_ma(ma_list, df["Temp"], 50)
    print(even_df)

    utils.print_title("ADF Calculations")
    ma = utils.get_moving_average(3, df["Temp"])
    ma = list(map(lambda x: x if not math.isnan(x) else 0, ma))
    utils.ADF_Cal(df["Temp"], "Original Data ADF")
    utils.ADF_Cal(df["Temp"] - ma, "3 MA ADF")

    utils.print_title("STL Decomposition")
    STL = STL(Temp, period = 365)
    res = STL.fit()
    fig = res.plot()

    T = res.trend
    S = res.seasonal
    R = res.resid

    plt.figure(figsize=(10, 6))
    plt.plot(T, label="Trend")
    plt.plot(S, label="Seasonality")
    plt.plot(R, label="Residue")
    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.title("STL Decomposition Graph - Trend, Season, Residue vs Time")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["Temp"], label="Original Data")
    plt.plot(df["Temp"] - S, label="Seasonally Adjusted Data")
    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.title("Seasonally Adjusted Data vs Time")
    plt.legend()
    plt.show()

    utils.print_title("Strength of Trend and Seasonality")
    utils.cal_strength(T, S, R)
