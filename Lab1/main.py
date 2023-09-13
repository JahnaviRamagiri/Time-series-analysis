
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


import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import utils

if __name__ == '__main__':
    file_path = utils.get_file_path("tute1.csv")

    # Question 1
    # Plot Sales, AdBudget and GPD versus time step in one graph. Add grid and appropriate
    # title, legend to each plot. The x-axis is the time, and it should show the time (year).
    pd.read_csv(file_path)
    



