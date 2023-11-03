import pandas as pd

from LAB4 import utils

if __name__ == '__main__':
    print()
    train_set = [112, 118, 132, 129, 121, 135, 148, 136, 119]
    # train_set = [39,44,40,45,38,43,39]
    test_set = [104, 118, 115, 126, 141]

    data = {'Method': [], 'Prediction MSE': [], 'Forecast MSE': [], 'Prediction Variance': [], 'Forecast Variance': [], 'Q Value': []}
    metrics_df = pd.DataFrame(data)
    title = ["Average", "Naive", "Drift", "SES"]

    # Average Method
    utils.print_title("Average Method")

    train_pred, test_forecast = utils.cal_avg_method(train_set, test_set)
    utils.plot_forecast(train_set, test_set, test_forecast, "Average Method")
    metrics_avg = utils.cal_metrics(train_set, test_set, train_pred, test_forecast, title[0])
    method = [title[0]]
    method.extend(metrics_avg)
    metrics_df.loc[len(metrics_df)] = method


    # Naive Method
    utils.print_title("Naive Method")
    train_pred, test_forecast = utils.cal_naive_method(train_set, test_set)
    utils.plot_forecast(train_set, test_set, test_forecast, "Naive Method")
    metrics_naive = utils.cal_metrics(train_set, test_set, train_pred, test_forecast, title[1])
    method = [title[1]]
    method.extend(metrics_naive)
    metrics_df.loc[len(metrics_df)] = method

    # Drift Method
    utils.print_title("Drift Method")
    train_pred, test_forecast = utils.cal_drift_method(train_set, test_set)
    utils.plot_forecast(train_set, test_set, test_forecast, "Drift Method")
    metrics_drift = utils.cal_metrics(train_set, test_set, train_pred, test_forecast, title[2])
    method = [title[2]]
    method.extend(metrics_drift)
    metrics_df.loc[len(metrics_df)] = method

    # SES Method
    utils.print_title("SES Method")
    train_pred, test_forecast = utils.cal_ses(train_set, test_set, 0.5)
    utils.plot_forecast(train_set, test_set, test_forecast, "SES Method")
    metrics_ses = utils.cal_metrics(train_set, test_set, train_pred, test_forecast, title[3])
    method = [title[3]]
    method.extend(metrics_ses)
    metrics_df.loc[len(metrics_df)] = method

    alphas = [0, 0.25, 0.75, 0.99]
    utils.plot_ses(train_set, test_set, alphas)

    utils.print_title("Evaluation Metrics")
    print(metrics_df.to_string(index=False))





