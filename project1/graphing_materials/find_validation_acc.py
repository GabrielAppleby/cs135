import pandas as pd


def main():
    cross_val_accs = pd.read_csv('accuracies_cross_val.csv')
    cross_val_accs = cross_val_accs.drop(cross_val_accs.columns[0], axis=1)
    print(cross_val_accs.mean())
    print(cross_val_accs.std())


if __name__ == '__main__':
    main()
