import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LAB4 import utils
from pandas_datareader import data
import yfinance as yf


if __name__ == '__main__':
    np.random.seed(6313)

    # Question 2
    mean = 0
    std = 1
    N = 1000
    white_noise = np.random.normal(mean, std, size = N)

    y = np.zeros(len(white_noise))
    date = pd.date_range(start = '2000-01-01',
                         end = '2000-12-31',
                         periods = len(y))
    df = pd.DataFrame(data = white_noise , columns=['y'], index = date)
    print(df.head())

    df.plot()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('White Noise')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(white_noise, bins=30, density=True, alpha=0.7, color='blue')
    plt.title('Histogram of White Noise')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    sampled_mean = np.mean(white_noise)
    sampled_std_dev = np.std(white_noise)

    print(f"Sampled Mean: {sampled_mean:.2f}")
    print(f"Sampled Standard Deviation: {sampled_std_dev:.2f}")

    # Question 2
    a = [3, 9, 27, 81,243]
    b = a[::-1]
    c = b + a[1:]

    # ACF for Makeup dataset
    n = len(a)
    max_lag = 4
    lags = np.arange(-max_lag, max_lag + 1)
    utils.auto_corr_func(a, lags, "ACF for Custom Input")

    # ACF for White Noise
    n = len(white_noise)
    max_lag = 20
    lags = np.arange(-max_lag, max_lag + 1)
    utils.auto_corr_func(white_noise, lags, "ACF for White Noise")

    # Question 4a
    yf.pdr_override()
    stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
    fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
    for ind, stk in enumerate(stocks):
        row = ind // 2
        col = ind % 2
        ax = axes[row, col]
        df_stk = data.get_data_yahoo(stk, start = '2000-01-01', end = '2023-01-31')
        ax.plot(df_stk["Close"])
        ax.set_ylabel('Magnitude')
        ax.set_xlabel(stk)
        ax.set_title(f'Close value for {stk}')

    plt.show()

    # Question 4b
    fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)

    for ind, stk in enumerate(stocks):
        print("Calculating ACF for ",stk)
        row = ind // 2
        col = ind % 2
        ax = axes[row, col]
        df_stk = data.get_data_yahoo(stk, start = '2000-01-01', end = '2023-01-31')

        n = len(df_stk["Close"])
        max_lag = 50
        lags = np.arange(-max_lag, max_lag + 1)
        acf = utils.auto_corr_func(df_stk["Close"], lags, f"ACF for {stk}", False)
        print(acf)

        (markers, stemlines, baseline) = ax.stem(lags, acf, markerfmt='o')
        plt.setp(markers, color = 'red', marker = 'o')
        plt.setp(baseline, color='grey', linewidth=2, linestyle='-')
        m = 1.96/np.sqrt(n)
        ax.axhspan(-m, m,alpha = 0.2, color = 'blue')
        ax.set_ylabel('Magnitude')
        ax.set_xlabel("Lags")
        ax.set_title(f"ACF for {stk}")

    plt.show()
