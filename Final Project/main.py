import pandas as pd
import utils

if __name__ == '__main__':
    file_path = utils.get_file_path("AirQuality.csv")

    utils.print_title("Load Data")
    df = pd.read_csv(file_path, delimiter=';')


    # date_range = pd.date_range(start='10/03/2004', periods=len(df), freq='h')
    # df = pd.DataFrame(df, index=date_range)
    # df = df.dropna()
    print(df.head())