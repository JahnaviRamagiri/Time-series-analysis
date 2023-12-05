import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from modules import statistics as st, plot, utils
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    utils.print_title("Dataset Description".upper(), "*", 30, 2)
    file_path = utils.get_file_path("AirQuality.csv")

    utils.print_title("Load Data", "~")
    target_name = "T"
    print(f"Target Variable : {target_name}")
    df = pd.read_csv(file_path, delimiter=';', decimal=",")
    date_range = pd.date_range(start='2004-03-10 18:00:00', periods=len(df), freq='h')
    df.index = date_range
    df = df[800:5800]
    target = df[target_name]

    utils.print_title("Holt Winter")
    train, test = train_test_split(target, shuffle=False, test_size=0.2)
    holtt = ets.ExponentialSmoothing(train, trend='mul', damped=True, seasonal='mul').fit()
    holtf = holtt.forecast(steps=len(test))
    holtf = pd.DataFrame(holtf).set_index(test.index)

    MSE = np.square(np.subtract(test.values, np.ndarray.flatten(holtf.values))).mean()
    print("Mean square error for holt-winter method is ", MSE)
    fig, ax = plt.subplots()
    ax.plot(train, label="train")
    ax.plot(test, label="test")
    ax.plot(holtf, label="Holt-Winter Method")

    plt.legend(loc='upper left')
    plt.title('Holt-Winter Method')
    plt.xlabel('Time (monthly)')
    plt.ylabel('Temperature')
    plt.show()
