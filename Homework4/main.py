import pandas as pd
from Lab1 import utils

if __name__ == '__main__':
    # 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡)
    utils.print_title("AR")
    num = [1, 0, 0]
    den = [1, -0.5, -0.2]

    df_mv = pd.DataFrame()
    df_mv.index = [100, 1000, 10000]

    y_100 = utils.get_arma(num, den, 100, 2, 1, "AR 100")
    y_1000 = utils.get_arma(num, den, 1000, 2, 1, "AR 1000")
    y_10000 = utils.get_arma(num, den, 10000, 2, 1, "AR 10000")
    ar_mean, ar_var = utils.get_mean_var([y_100, y_1000, y_10000])
    df_mv["AR Mean"] = ar_mean
    df_mv["AR Var"] = ar_var

    acf = []
    lags = [20, 40, 80]
    for lag in lags:
        acf.append(utils.auto_corr_func(y_100, lag, title=f"AR ACF Lag: {lag}", plot=True))


    # 𝑦(𝑡) = 𝑒(𝑡) + 0.1𝑒(𝑡 − 1) + 0.4𝑒(𝑡 − 2)
    utils.print_title("MA")
    num = [1, 0.1, 0.4]
    den = [1, 0, 0]

    y_100 = utils.get_arma(num, den, 100, 2, 1, "MA 100")
    y_1000 = utils.get_arma(num, den, 1000, 2, 1, "MA 1000")
    y_10000 = utils.get_arma(num, den, 10000, 2, 1, "MA 10000")
    ma_mean, ma_var = utils.get_mean_var([y_100, y_1000, y_10000])
    df_mv["MA Mean"] = ma_mean
    df_mv["MA Var"] = ma_var

    lags = [20, 40, 80]
    for lag in lags:
        acf.append(utils.auto_corr_func(y_100, lag, title=f"MA ACF Lag: {lag}", plot=True))

    # 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) − 0.2𝑦(𝑡 − 2) = 𝑒(𝑡) + 0.1𝑒(𝑡 − 1) + 0.4𝑒(𝑡 − 2)
    utils.print_title("ARMA")
    num = [1, 0.1, 0.4]
    den = [1, -0.5, -0.2]

    y_100 = utils.get_arma(num, den, 100, 2, 1, "ARMA 100")
    y_1000 = utils.get_arma(num, den, 1000, 2, 1, "ARMA 1000")
    y_10000 = utils.get_arma(num, den, 10000, 2, 1, "ARMA 10000")
    arma_mean, arma_var = utils.get_mean_var([y_100, y_1000, y_10000])
    df_mv["ARMA Mean"] = arma_mean
    df_mv["ARMA Var"] = arma_var

    lags = [20, 40, 80]
    for lag in lags:
        acf.append(utils.auto_corr_func(y_100, lag, title=f"ARMA ACF Lag: {lag}", plot=True))

    theoritical_val = {
    'AR Mean': 6.67,
    'AR Var': 1.71,
    'MA Mean': 3.0,
    'MA Var': 1.17,
    'ARMA Mean': 10,
    'ARMA Var': 3
    }

    new_df = pd.DataFrame([theoritical_val], index=['theoritical'])
    df_mv = pd.concat([df_mv, new_df])

    utils.print_title("Mean - Variance Theoritical - Practical Value Comparison")
    df_string = df_mv.to_string()
    print(df_string)





