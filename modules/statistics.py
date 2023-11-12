import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import warnings
import plot
import utils

warnings.filterwarnings("ignore")

def get_statistics(df, col_list):
    """

    :param df: DataFrame
    :param col_list: List of Columns
    :return: null

    Calculates mean, variance and standard deviation of the dataframe
    """
    mean = np.mean(df)
    var = np.var(df)
    std = np.std(df)
    median = df.median()

    for ind, col in enumerate(col_list[1:]):
        print(f"The {col} mean is: {mean[ind]:.2f} and the variance is {var[ind]:.2f} with standard deviation {std[ind+1]:.2f} median:{median[ind]:.2f}")


def get_rolling_mean(df_col):
    """

    :param df_col
    :return: Rolling Mean
    """
    acc_sum = 0
    roll_mean = []
    for ind, i in enumerate(df_col):
        acc_sum += i
        roll_mean.append(acc_sum/(ind+1))
    return roll_mean


def get_rolling_var(df_col):
    """

    :param df_col:
    :return: Rolling Variance
    """
    roll_var = []
    for ind, i in enumerate(df_col):
        df_var = df_col.head(ind).var()
        roll_var.append(df_var)

    return roll_var


def cal_rolling_mean_var(df_col, title):
    r_mean = get_rolling_mean(df_col)
    r_var = get_rolling_var(df_col)

    plot.plot_rolling_mean_var(r_mean, r_var, title)
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


def get_mean_var(y_list):
    return [np.mean(y) for y in y_list], [np.var(y) for y in y_list]


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

# standardize dataset
def standardize_dataset(dataset):
    means = utils.column_means(dataset)
    stdevs = utils.column_stdevs(dataset, means)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
