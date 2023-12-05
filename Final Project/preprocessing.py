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

    utils.get_df_info(df, "Original Dataset")
    print(df.describe())
    utils.print_observation(f"The Dataset contains {df.shape[0]} Records and {df.shape[1]} Columns. "
                            f" We need to further preprocess the data.")

    utils.print_title("Preprocessing Dataset")
    date_range = pd.date_range(start='2004-03-10 18:00:00', periods=len(df), freq='h')
    df.index = date_range
    utils.print_title("Handling Null values", "~")
    print(df.isna().sum())

    utils.print_title("Missing Values", "~")
    # TODO: Find column statistics
    # Dropping unnamed columns 15 and 16
    df.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
    utils.get_df_info(df, "Dataset after dropping Extra columns")
    df.dropna(inplace=True)
    print(df.isna().sum())

    utils.print_title("Plot Target Variable", "~")
    plot.plot_variable_graph(df, target_name, "Time", f"{target_name} ", f"{target_name} vs Time ", 200)
    utils.print_observation(
        "The Data Shows sudden spikes to -200, as the missing values in the dataset are represented by -200. These need to handled.")

    df.replace(to_replace=-200, value=np.nan, inplace=True)
    utils.print_title("Plot Target Variable", "~")
    plot.plot_graph(df, df.columns, "Time", 1, "Features", "Features vs Time", 200)
    # plot.plot_variable_graph(df, target_name, "Time", f"{target_name} in mg/m^3", f"{target_name} vs Time", 200)

    utils.print_title("Plot Missing Values", "~")
    plt.figure(figsize=(12, 15))
    sns.heatmap(df.isna(), yticklabels=False, cmap='crest')
    plt.show()
    utils.print_observation("Column NMHC(GT) has the most number of null values - Dropping NHMC")
    df.drop(columns=['NMHC(GT)'], inplace=True)
    print("Missing Values")
    print(df.isna().sum())

    print("Replacing missing values with their mean")
    for i in df.columns:
        df[i] = df[i].fillna(df[i].mean())
    print(df.isna().sum())

    df = df[800:5800]

    df.to_csv("Preprocessed Dataset.csv")