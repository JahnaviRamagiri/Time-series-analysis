import pandas as pd
from LAB4 import utils
import statsmodels.api as sm

if __name__ == '__main__':
    url = "https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/blob/main/weight-height.csv"
    file_path = utils.get_file_path("weight-height.csv")
    df = pd.read_csv(file_path)
    print(df.head())

    df["Gender"] = df["Gender"].map({"Male" : 1, "Female" : 2})

    weight = df["Weight"]
    height = df["Height"]
    gender = df["Gender"]

    utils.print_title("Correlation Coefficient")
    rwh = utils.correlation_coefficent_cal(weight, height)
    rhg = utils.correlation_coefficent_cal(gender, height)
    rwg = utils.correlation_coefficent_cal(weight, gender)
    print(rwh, rhg, rwg)

    rwh_g = utils.cal_partial_correlation(rwh, rhg, rwg)
    print(rwh_g)

    utils.print_title("Statistical Significance")
    # t0 =





