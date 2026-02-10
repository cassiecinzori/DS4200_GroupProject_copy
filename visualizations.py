from api311 import Year
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    year15 = Year("data/cleaned2015.csv")
    year25 = Year("data/cleaned2025.csv")
    year15.make_points()
    year25.make_points()

    summary_type_dct = year15.summarize("type")

    print(summary_type_dct.keys())
    print(summary_type_dct["describe"])
    monthly_type_data = summary_type_dct["monthly"]
    print(summary_type_dct["val_counts"])
    print(monthly_type_data.head())


if __name__ == "__main__":
    main()
