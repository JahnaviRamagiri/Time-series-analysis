import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from LAB4 import utils
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

if __name__ == '__main__':

    # Using Pandas library load the time series dataset from the BB. Split the dataset into training set
    # and test set. Use 80% for training and 20% for testing. Display the size of the train set and test set
    # on the console
    file_path = utils.get_file_path("autos.clean.csv")
    df = pd.read_csv(file_path)
    print(df.head())
    features = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg","highway-mpg"]

    y = df["price"]
    x = df[features]
    print(x.info())
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, shuffle=False, test_size=0.2)
    print(f"Size of Train set = {xtrain.shape}, Size of Test set {ytrain.shape}")

    df_new = x
    df_new["price"] = df["price"]

    # Plot the correlation matrix of dependent and independent variables using the seaborn package
    # and heatmap function. Write down your observations about the correlation matrix
    df_corr = df_new.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Collinearity detection
    # Perform SVD analysis on the original feature space and write down your observation if colinearity exists.
    U, S, V = np.linalg.svd(df_new)
    print("Singular value of x", S)


    # Calculate the condition number
    print("Condition Number of x", np.linalg.cond(x))

    # Standardize the Dataset
    # utils.standardize_dataset(df_new)
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df_new), columns=df_new.columns)
    y = standardized_df["price"]
    x = standardized_df[["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg","highway-mpg"]]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, shuffle=False, test_size=0.2)

    # Estimate Beta LSE
    # B = ((XTX)^-1)*XT*Y
    xtrain = sm.add_constant(xtrain)
    beta_hat = np.linalg.inv(xtrain.T @ xtrain) @ xtrain.T @ ytrain
    print("Beta Hat", '\n', beta_hat)


    # OLS Function
    model = sm.OLS(ytrain, xtrain).fit()
    print(model.summary())

    # Backward Step Regression
    aic = []
    bic = []
    adj_r2 = []
    p = []
    drop = []
    vif_max = []

    target_variable = "price"
    selected_features = features

    while len(selected_features) > 1:
        model, p_values = utils.get_OLS(standardized_df, selected_features, target_variable)
        aic.append(model.aic)
        bic.append(model.bic)
        adj_r2.append(model.rsquared_adj)
        # p.append(p_values)
        drop.append(selected_features[p_values.argmax()])

        print(f"Dropping {selected_features[p_values.argmax()]} with highest p-value {p_values.max()}")
        selected_features.remove(selected_features[p_values.argmax()])

    df_bsr = pd.DataFrame()
    df_bsr["AIC"] = aic
    df_bsr["BIC"] = bic
    df_bsr["Adjusted R2"] = adj_r2
    df_bsr["Least Important Feature"] = drop

    print(df_bsr)
    df_bsr.to_csv("Backward Step Regression.csv")


    # Reduction of VIF
    aic = []
    bic = []
    adj_r2 = []
    p = []
    drop = []

    target_variable = "price"

    features = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore",
                "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"]
    selected_features = features

    while len(standardized_df.columns) > 1:
        model, p_values = utils.get_OLS(standardized_df, standardized_df.columns, target_variable)

        # p.append(p_values)
        vif_series = pd.Series([variance_inflation_factor(standardized_df.values, i) for i in range(standardized_df.shape[1])],
                               index=standardized_df.columns)
        feature_to_remove = vif_series.idxmax()
        if feature_to_remove == "price":
            feature_to_remove = vif_series.drop("price").idxmax()

        aic.append(model.aic)
        bic.append(model.bic)
        adj_r2.append(model.rsquared_adj)
        drop.append(feature_to_remove)
        vif_max.append(vif_series.max())

        print(f"Dropping {feature_to_remove} with highest vif-value {vif_series.max()}")
        standardized_df = standardized_df.drop(feature_to_remove, axis = 1)

    df_vif = pd.DataFrame()
    df_vif["AIC"] = aic
    df_vif["BIC"] = bic
    df_vif["Adjusted R2"] = adj_r2
    df_vif["Least Important Feature"] = drop
    df_vif["VIF (Dropped Feature)"] = vif_max

    print(df_vif)
    df_vif.to_csv("Reduction with VIF.csv")



    # Final Features
    final_features = []
    final_model, p_values = utils.get_OLS(standardized_df, final_features, target_variable)
    predictions = model.predict(xtest)

    # Plot the training, test, and predicted values
    plt.figure(figsize=(8, 6))
    plt.plot(train_df['X1'], Y_train, label='Train Data', marker='o', linestyle='-')
    plt.plot(test_df['X1'], test_df['Y'], label='Test Data', marker='o', linestyle='-')
    plt.plot(test_df['X1'], Y_test_predicted, label='Predicted Data', marker='o', linestyle='--')

    # Add labels and a legend
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.title('Train, Test, and Predicted Data')
    plt.legend()

    # Show the plot
    plt.show()





        # The 'standardized_data' variable now contains your standardized dataset

    # vif_data1 = pd.DataFrame()
    # vif_data1["features"] = df_new.columns
    # vif_data1["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    # print(vif_data1)


