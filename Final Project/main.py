import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from modules import statistics as st, plot, utils
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split

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

    utils.print_title("Loading Preprocessed Dataset")
    df = pd.read_csv("Preprocessed Dataset.csv", index_col = 0)
    target_name = "T"
    target = df[target_name]

    # Analysis on Target Variable
    print("Target Variable ", target_name)
    plot.plot_variable_graph(df, target_name, "Time", f"{target_name}", f"{target_name} vs Time After Handling Missing Values", 200)

    utils.print_title("Plot Auto Correlation function", "~")
    target_acf = st.auto_corr_func(target, 25, title=f"ACF {target_name}", plot=True)
    print(target_acf)
    # TODO: Write ACF Observations.

    utils.print_title("Correlation Matrix", "~")
    utils.plot_corr_matrix(df)
    # TODO : Write Correlaion observations

    # utils.print_title("Train-Test Split", "~")
    # split_idx = 0.8 * len(df)
    # train_set = df.iloc[:split_idx]
    # test_set = df.iloc[split_idx:]
    # print(train_set.shape, test_set.shape)

    utils.print_title("Stationarity".upper(), "*", 30, 2)
    utils.check_stationarity(target, f"Stationarity Test on {target_name}", order=0)

    df["Log Transform"] = utils.get_log_transform(target)
    df["Seasonal Differencing"] = st.get_differencing(df["Log Transform"], True, 24)
    st.ACF_PACF_Plot(df["Seasonal Differencing"][24:], 20)
    utils.check_stationarity(df["Seasonal Differencing"], "Seasonal Differencing", order=24)
    st.auto_corr_func(df["Seasonal Differencing"], 25, title=f"ACF First Order Differencing {target_name}", plot=True)
    plot.plot_variable_graph(df, "Seasonal Differencing", "Time", f"{target_name}",
                             f"{target_name} vs Time After Seasonal Differencing", len(df))
    st.ACF_PACF_Plot(df["Seasonal Differencing"][24:], 20)

    df["First Diff"] = st.get_differencing(df["Seasonal Differencing"], ns_period=1)
    utils.check_stationarity(df["First Diff"], "First Order Differencing", order=25)
    st.auto_corr_func(df["First Diff"], 25, title=f"ACF First Order Differencing {target_name}", plot=True)
    plot.plot_variable_graph(df, "First Diff", "Time", f"{target_name}",
                             f"{target_name} vs Time After First Differencing", len(df))
    st.ACF_PACF_Plot(df["First Diff"][25:], 25)


    # STL Decomposition
    utils.print_title("STL Decomposition")


