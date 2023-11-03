from LAB4 import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    z = [-1, -2, -3, -4, -5]
    g = [1, 1, 0, -1, -1, 0, 1]
    h = [0, 1, 1, 1, -1, -1, -1]

    # The correlation coefficient between x and y. Display the answer on the console.
    rxy = utils.correlation_coefficent_cal(x, y)
    rxz = utils.correlation_coefficent_cal(x, z)
    rgh = utils.correlation_coefficent_cal(g, h)
    print(f"The correlation between x and y is {rxy:.2f}")
    print(f"The correlation between x and z is {rxz:.2f}")
    print(f"The correlation between g and h is {rgh:.2f}")

    file_path = utils.get_file_path("tute1.csv")
    df = pd.read_csv(file_path)
    print(df.head())

    rsg = utils.correlation_coefficent_cal(df["Sales"], df["GDP"])
    df.plot.scatter(x='Sales', y = 'GDP')
    plt.title(f'Scatter Plot of Sales and GDP coef = {rsg: .2f}')
    plt.show()
    print(f"The correlation between Sales and GDP is {rsg: .2f}")

    rsa = utils.correlation_coefficent_cal(df["Sales"], df["AdBudget"])
    df.plot.scatter(x='Sales', y='AdBudget')
    plt.title(f'Scatter Plot of Sales and AdBudget coef = {rsa: .2f}')
    plt.show()
    print(f"The correlation between Sales and AdBudget is {rsa: .2f}")

    rga = utils.correlation_coefficent_cal(df["GDP"], df["AdBudget"])
    df.plot.scatter(x='GDP', y='AdBudget')
    plt.title(f'Scatter Plot of GDP and AdBudget coef = {rga: .2f}')
    plt.show()
    print(f"The correlation between GDP and AdBudget is {rga: .2f}")

    sns.pairplot(df, kind="kde")
    plt.show()
    sns.pairplot(df, kind="hist")
    plt.show()
    sns.pairplot(df, diag_kind="hist")
    plt.show()

    df_corr = df.corr()
    sns.heatmap(df_corr)
    plt.show()

    print("Input parameters for generating random variable")
    mean = float(input("Input Mean") or 0)
    variance = float(input("Input Variance") or 1)
    observations = int(input("Input Observations") or 1000)

    np.random.seed(6313)
    x = np.random.normal(mean, np.sqrt(variance), observations)
    y = x**2
    z = x**3

    rxy = utils.correlation_coefficent_cal(x, y)
    print(f"The correlation between x and y is {rxy:.2f}")

    rxz = utils.correlation_coefficent_cal(x, z)
    print(f"The correlation between x and z is {rxz:.2f}")

    # Create scatter plots to visualize the relationships
    plt.figure(figsize=(12, 5))

    # Plot x vs. y
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.5)
    plt.title('Scatter Plot of x vs. y')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot x vs. z
    plt.subplot(1, 2, 2)
    plt.scatter(x, z, alpha=0.5)
    plt.title('Scatter Plot of x vs. z')
    plt.xlabel('x')
    plt.ylabel('z')

    plt.tight_layout()
    plt.show()
