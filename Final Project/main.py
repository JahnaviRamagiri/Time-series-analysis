import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from modules import statistics as st, plot, utils

if __name__ == '__main__':
    """
    Dataset Description
    Stationarity
    Time Series Decomposition
    Holt - Winter’s Method
    Feature Selection
    Base Models - Simple Forecasting
    Mulitple Linear Regression Model
    Time Series Model
    Forecast Function
    Residual Analysis and Diagnostic Test
    Deep Learning Model
    Final Model Selection
    Model Prediction
    """

    utils.print_title("Dataset Description".upper(), "*", 30, 2)
    file_path = utils.get_file_path("AirQuality.csv")

    utils.print_title("Load Data", "~")
    print(f"Target Variable : CO in mg/m^3")
    df = pd.read_csv(file_path, delimiter=';', decimal=",")
    utils.get_df_info(df, "Original Dataset")
    # print(df.info())
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
    # utils.print_observation(
    df.drop(['Date', 'Time','Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
    utils.get_df_info(df, "Dataset after dropping Extra columns")
    df.dropna(inplace=True)
    print(df.isna().sum())

    utils.print_title("Plot Target Variable", "~")
    plot.plot_variable_graph(df, "CO(GT)", "Time", "CO(GT) in mg/m^3", "CO(GT) vs Time", 200)
    utils.print_observation("The Data Shows sudden spikes to -200, as the missing values in the dataset are represented by -200. These need to handled.")

    df.replace(to_replace=-200, value=np.nan, inplace=True)
    # utils.print_title("Plot Target Variable", "~")
    plot.plot_graph(df, df.columns, "Time", 1, "Features", "Features vs Time", 200)
    plot.plot_variable_graph(df, "CO(GT)", "Time", "CO(GT) in mg/m^3", "CO(GT) vs Time", 200)

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
    plot.plot_variable_graph(df, "CO(GT)", "Time", "CO(GT) in mg/m^3", "CO(GT) vs Time After Handling Missing Values", 200)

    utils.print_title("Plot Auto Correlation function", "~")
    target_acf = st.auto_corr_func(df["CO(GT)"], 25, title="ACF CO(GT)", plot=True)
    print(target_acf)
    # TODO: Write ACF Observations.

    utils.print_title("Correlation Matrix", "~")
    df_corr = df.corr()
    plt.figure(figsize=(10, 12))
    sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
    plt.title("Dataset Correlation Heatmap")
    plt.show()
    # TODO : Write Correlaion observations

    # utils.print_title("Train-Test Split", "~")
    # split_idx = 0.8 * len(df)
    # train_set = df.iloc[:split_idx]
    # test_set = df.iloc[split_idx:]
    # print(train_set.shape, test_set.shape)

    utils.print_title("Stationarity".upper(), "*", 30, 2)
    st.ADF_Cal(df["CO(GT)"], "CO(GT)")
    st.kpss_test(df["CO(GT)"], "CO(GT)")
