
"""
LAB 1
Submission guidelines:


A. The softcopy of the developed Python code .py must also be submitted separately. Please make
sure the developed python code runs without any error by testing it through PyCharm software.
The developed python code with any error will subject to 50% points penalty.
B. Add an appropriate x-label, y-label, legend, and title to each graph.
C. Write a report and answer all the above questions. Include the required graphs in your report.
D. Submission: report (pdf format) + .py . The python file is a supporting file and will not replace the
solution. A report that includes the solution to all questions is required and will be graded.
E. The python file must regenerate the provided results inside the report.
"""

import pandas as pd
from LAB4 import utils

if __name__ == '__main__':
    file_path = utils.get_file_path("tute1.csv")

    # Question 1: Plot Sales, AdBudget and GPD versus time step in one graph. Add grid and appropriate
    # title, legend to each plot. The x-axis is the time, and it should show the time (year).
    df = pd.read_csv(file_path)
    print(df.head())
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    utils.plot_graph(df, df.columns, "Date", 1, "USD($)", "Sales, AdBudget, GDP vs Time")

    # Question 2: Find the time series statistics (average, variance, standard deviation, median) of Sales, AdBudget
    # and GPD
    utils.get_statistics(df, df.columns)

    # Question 3: Prove that the Sales, AdBudget and GDP in this time series dataset is stationary.
    utils.cal_rolling_mean_var(df["Sales"], "Sales")
    utils.cal_rolling_mean_var(df["AdBudget"], "AdBudget")
    utils.cal_rolling_mean_var(df["GDP"], "GDP")

    # Question 5: Perform an ADF-test to check if the Sales, AdBudget and GDP stationary or not
    thres = 0.95
    utils.ADF_Cal(df["Sales"], "Sales", thres)
    utils.ADF_Cal(df["AdBudget"], "AdBudget", thres)
    utils.ADF_Cal(df["GDP"], "GDP", thres)

    # Question 6: Perform an KPSS-test to check if the Sales, AdBudget and GDP stationary or not
    utils.kpss_test(df["Sales"], "Sales", thres)
    utils.kpss_test(df["AdBudget"], "AdBudget", thres)
    utils.kpss_test(df["GDP"], "GDP", thres)

    # Question 7: "AirPassengers.csv
    file_path = utils.get_file_path("../In-class/AirPassengers.csv")
    df = pd.read_csv(file_path)
    print(df.head())
    df["Month"] = pd.to_datetime(df["Month"], format = "%Y-%m")
    utils.plot_graph(df, df.columns, "Month", 20, "#Passengers", "#Passengers vs Time")
    utils.get_statistics(df, df.columns)
    utils.check_stationarity(df["#Passengers"], "#Passengers", thres=0.05)

    # Question 8
    df["First Diff"] = utils.get_differencing(df["#Passengers"])
    utils.check_stationarity(df["First Diff"], "First Order Differencing", thres=0.05, order=1)
    df["Second Diff"] = utils.get_differencing(df["First Diff"])
    utils.check_stationarity(df["Second Diff"], "Second Order Differencing", thres=0.05, order=2)
    df["Third Diff"] = utils.get_differencing(df["Second Diff"])
    utils.check_stationarity(df["Third Diff"], "Third Order Differencing", thres=0.05, order=3)

    # Log Transform
    df["Log Transform"] = utils.get_log_transform(df["#Passengers"])
    df["First Diff Log Transform"] = utils.get_differencing(df["Log Transform"])
    utils.check_stationarity(df["First Diff Log Transform"], "First Diff Log Transform", thres=0.05, order = 1)

