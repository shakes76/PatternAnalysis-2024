import argparse
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    metadata = pd.read_csv(args.input)


def split_sub_population(df: pd.DataFrame, id_name: str) -> tuple:
    df_train = df.sample(random_state=42, frac=0.8)
    df_test = df[~df[id_name].isin(df_train[id_name])]

    return df_train, df_test

if __name__ == "__main__":
    main()