import pandas as pd

def main():

    training_df = pd.read_csv("data/melting-point/train.csv")
    print(training_df.head())

if __name__ == "__main__":
    main()