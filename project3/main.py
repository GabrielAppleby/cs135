import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV


def main() -> None:
    """
    Gets the confidence of the best classifier, and produces graphs of some
    of the tests on all of the classifiers and data representations.
    :return: Nothing
    """
    train_data: np.array = read_data('data/trainset.csv')
    param_grid = {'n_factors': [5, 10, 20, 40, 80, 160, 320],
                  'reg_all': [.02, .04, .08, .16, .32, .64, 1.28, 2.56, 5.12]}
    #trainset, testset = train_test_split(train_data, test_size=.25)

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(train_data)

    results_df = pd.DataFrame.from_dict(gs.cv_results)

    results_df.to_csv('results.csv')
    #cross_validate(algo, train_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # df = pd.read_csv('results.csv')
    # df = df[['mean_test_rmse', 'std_test_rmse', 'param_n_factors', 'param_reg_all']]
    # print(df.to_string())

def read_data(file_name: str) -> np.array:
    """
    Reads in the data from the train reviews.
    :param file_name: The name of the file to read in.
    :return: The data np array.
    """
    df: pd.DataFrame = pd.read_csv(file_name)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[list(df)], reader)
    return data


if __name__ == '__main__':
    main()
