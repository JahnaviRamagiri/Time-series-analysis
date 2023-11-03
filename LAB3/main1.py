import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from LAB4 import utils
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


if __name__ == '__main__':

    # Question 1
    utils.print_title("LOAD DATA")
    file_path = utils.get_file_path("autos.clean.csv")
    df = pd.read_csv(file_path)
    print(df.head())
    features = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg","highway-mpg"]

    y = df["price"]
    x = df[features]
    print(x.info())
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, shuffle=False, test_size=0.2)
    print(f"Size of Train set = {xtrain.shape}, Size of Test set {xtest.shape}")

    df_new = x
    df_new["price"] = df["price"]

    # Question 2
    df_corr = df_new.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Collinearity Detection
    utils.print_title("COLLINEARITY DETECTION")
    U, S, V = np.linalg.svd(df_new)
    print("Singular value of x", S)
    # Condition Number
    print(f"Condition Number of x {np.linalg.cond(x):.2f}")

    # STANDARDIZATION
    utils.print_title("STANDARDIZATION")
    features = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg","highway-mpg"]
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
    y = standardized_df["price"]
    x = standardized_df[["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg","highway-mpg"]]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, shuffle=False, test_size=0.2)

    # Estimate Beta LSE
    xtrain_ = sm.add_constant(xtrain)
    beta_hat = np.linalg.inv(xtrain_.T @ xtrain_) @ xtrain_.T @ ytrain
    print("Beta Hat", '\n', beta_hat)

    # OLS Function
    model = sm.OLS(ytrain, xtrain).fit()
    print(model.summary())

    # Backward Step Regression
    utils.print_title("Backward Step Regression")
    bsr_best_model, bsr_features = utils.get_back_step_regression(xtrain, ytrain, features)

    # VIF
    utils.print_title("VIF Method")
    features = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore",
                "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"]
    vif_best_model, vif_features = utils.get_vif_reduction(xtrain, ytrain, features)


    # Final Model
    utils.print_title("FINAL OLS MODEL")
    final_features = bsr_features
    final_model, p_values = utils.get_OLS(xtrain, ytrain, final_features)
    print(final_model.summary())
    xtest_ = sm.add_constant(xtest[final_features])
    y_pred = final_model.predict(xtest_)
    y_tr_pred = final_model.predict(sm.add_constant(xtrain[final_features]))

    std = scaler.scale_[-1]
    mean = scaler.mean_[-1]

    # De-normalize y values
    y_tr = utils.inv_std(ytrain, std, mean)
    y_ts = utils.inv_std(ytest, std, mean)
    y_pr = utils.inv_std(y_pred, std, mean)

    # Plot the training, test, and predicted values
    plt.figure(figsize=(8, 6))
    plt.plot(y_tr, label='Train Data')
    plt.plot(y_ts, label='Test Data')
    plt.plot(y_pr, label='Predicted Data')

    # Add labels and a legend
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title('Price Prediction Plot Using OLS')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

    # Error Analysis
    utils.print_title("ERROR ANALYSIS")
    output = pd.DataFrame()
    output["Y"] = y_ts
    output["Ypred"] = y_pr
    output["Error"] = y_ts - y_pr
    acf = utils.auto_corr_func(output["Error"], 20, "Prediction Error ACF", plot=True)

    # # T-Test and F-Test
    utils.print_title("T-Test and F-Test")
    print(final_model.t_test(np.eye(len(final_model.params))))
    print(final_model.f_test(np.eye(len(final_model.params))))
    # t_test_results = final_model.t_test()
    # print("T-Test Results:")
    # print(t_test_results)
    #
    # f_test_result = final_model.f_test()
    # print("\nF-Test Result:")
    # print(f_test_result)









