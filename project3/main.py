import numpy as np
import pandas as pd

from collections import defaultdict
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate as sk_cross_val
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV as SkGridSearch
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from surprise import Dataset
from surprise import trainset
from surprise.dataset import DatasetAutoFolds
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV as SurGridSearch
from surprise.prediction_algorithms.predictions import Prediction
from surprise.model_selection import cross_validate as sur_cross_val
from typing import Dict, List, Iterator


def main() -> None:
    """
    Gets the confidence of the best classifier, and produces graphs of some
    of the tests on all of the classifiers and data representations.
    :return: Nothing
    """
    train_df: DataFrame = read_data('data/trainset.csv')
    test_df: DataFrame = read_data('data/testset.csv')
    genders_df: DataFrame = read_data('data/gender.csv')
    release_dates_df: DataFrame = read_data('data/release-year.csv')
    combined_df: DataFrame = pd.concat([train_df, test_df])

    train_dataset: DatasetAutoFolds = convert_to_dataset(train_df)
    test_dataset: DatasetAutoFolds = convert_to_dataset(test_df)
    combined_dataset: DatasetAutoFolds = convert_to_dataset(combined_df)

    train_set: trainset = train_dataset.build_full_trainset()
    combined_train_set: train_set = combined_dataset.build_full_trainset()

    test_list: List = test_dataset.build_full_trainset().build_testset()

    genders: np.array = genders_df.values.ravel()
    release_dates: np.array = release_dates_df.values.ravel()

    # grid_search_svd(train_dataset)
    # evaluate_mea(train_dataset)
    # evaluate_k_predictions(train_set, test_list)
    # grid_search_logistic(combined_train_set,
    #                      genders,
    #                      release_dates)
    evaluate_gender_and_year(combined_train_set, genders, release_dates)


def evaluate_gender_and_year(train_set: trainset,
                             genders: np.array,
                             release_dates: np.array):
    """
    
    :param train_set:
    :param genders:
    :param release_dates:
    :return:
    """
    svd: SVD = SVD()
    svd.fit(train_set)
    pipeline: Pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(solver='liblinear',
                                             multi_class='auto'))
    scores: np.array = sk_cross_val(
        pipeline, svd.pu, y=genders, cv=5, return_train_score=True)
    print("Train error: " + str(1 - np.mean(scores['train_score'])))
    print("Test error: " + str(1 - np.mean(scores['test_score'])))
    scores: np.array = sk_cross_val(
        pipeline, svd.qi, y=release_dates, cv=5, return_train_score=True)
    print("Train error: " + str(1 - np.mean(scores['train_score'])))
    print("Test error: " + str(1 - np.mean(scores['test_score'])))
    mean_release_year: float = np.mean(release_dates)
    print("Naive: " + str(
        1 - accuracy_score(release_dates, np.full_like(
            release_dates, mean_release_year))))


def grid_search_svd(train_data: Dataset) -> None:
    """
    Perform a grid search to find the best number of hyper parameters for SVD.
    :param train_data: The data set to use for the grid search.
    :return: Nothing.
    """
    # Perform grid search
    param_grid: Dict = {
        'n_factors': [5, 10, 20, 40, 80, 160, 320],
        'reg_all': [.02, .04, .08, .16, .32, .64, 1.28, 2.56, 5.12]}
    gs: SurGridSearch = SurGridSearch(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(train_data)

    # Best option
    print(gs.best_score['mae'])
    print(gs.best_params['mae'])

    # Save full results for later
    results_df: pd.DataFrame = pd.DataFrame.from_dict(gs.cv_results)
    results_df.to_csv('results.csv')

    # Print out other important info now
    results_df = results_df[['mean_test_mae',
                             'std_test_mae',
                             'param_n_factors',
                             'param_reg_all']]
    print(results_df.to_string())


def evaluate_mea(train_data: Dataset) -> None:
    """
    Get the mean absolute error for the SVD on the given data set.
    :param train_data: The data set to use.
    :return: Nothing.
    """
    sur_cross_val(SVD(), train_data, measures=['MAE'], cv=5, verbose=True)


def grid_search_logistic(train_set: trainset,
                         genders: np.array,
                         release_dates: np.array) -> None:
    """
    Perform a grid search to find the best number of hyper parameters for
    logistic regression.
    :param train_set: The train set to use to produce the user and item
    matrices.
    :param genders: The genders of the users.
    :param release_dates: The release dates of the movies.
    :return: Nothing.
    """
    svd: SVD = SVD()
    svd.fit(train_set)
    pipeline: Pipeline = make_pipeline(
        StandardScaler(), LogisticRegression())
    params: Dict = {'logisticregression__C': [.004, .008, .016, .032, .064],
                    'logisticregression__solver': ['liblinear'],
                    'logisticregression__penalty': ['l2'],
                    'logisticregression__warm_start': ['true'],
                    'logisticregression__multi_class': ['ovr']}

    gs_clf: SkGridSearch = SkGridSearch(
        pipeline,
        params,
        cv=2,
        iid=False,
        n_jobs=-1,
        return_train_score=True,
        error_score='raise')
    scored_params_gender: List = grid_search_logistic_helper(
        gs_clf, svd.pu, genders)
    print_helper(scored_params_gender)
    scored_params_year: List = grid_search_logistic_helper(
        gs_clf, svd.qi, release_dates)
    print_helper(scored_params_year)


def grid_search_logistic_helper(gs_clf, data, target) -> Iterator:
    """
    Run the grid search that is set up in the caller, zips up the mean test
    scores, std test scores, and params.
    :param gs_clf: The grid search to run.
    :param data: The data to fit on.
    :param target: The labels of the data to fit on.
    :return: The zip of the tuple, with mean test scores, test stds, and
    params.
    """
    gs_clf = gs_clf.fit(data, target)
    mean_test_scores = gs_clf.cv_results_['mean_test_score']
    test_stds = gs_clf.cv_results_['std_test_score']
    params = gs_clf.cv_results_['params']
    return zip(mean_test_scores, test_stds, params)


def print_helper(scored_params: List) -> None:
    """
    Prints the test scores and std deviations for each parameter combination.
    :param scored_params: The zip of the tuple, with mean test scores, test
    stds, and params.
    :return: Nothing
    """
    for test_mean, std, params in scored_params:
        print("Test: %0.3f (+/-%0.03f) for %r"
              % (test_mean, std, params))


def convert_to_dataset(df: DataFrame) -> DatasetAutoFolds:
    """
    Converts a dataframe into a surprise Dataset.
    :param df: The dataframe to convert.
    :return: The converted DataSet.
    """
    reader: Reader = Reader(rating_scale=(1, 5))
    data: DatasetAutoFolds = Dataset.load_from_df(df, reader)
    return data


def evaluate_k_predictions(train_set: trainset, test_list: List) -> None:
    """
    Evaluate the SVD on the given train set using the top k predictions.
    :param train_set: The train set use to evaluate the SVD.
    :param test_list: The test list used to evaluate the SVD.
    :return: Nothing.
    """
    svd: SVD = SVD()
    svd.fit(train_set)
    test_x: List = train_set.build_anti_testset()
    predictions: List[Prediction] = svd.test(test_x)
    prediction_dict: Dict = predictions_to_top_n(predictions)
    actual_dict: Dict = test_list_to_dict(test_list)
    total_rating: int = 0
    for user, predicted_ratings in prediction_dict.items():  # type: str, List
        actual_ratings: Dict = actual_dict.get(user, {})
        for predicted_rating in predicted_ratings:
            total_rating += actual_ratings.get(predicted_rating[0], 2)

    print(total_rating / (len(prediction_dict.keys()) * 5))


def read_data(file_name: str) -> np.array:
    """
    Reads in the data from the train reviews.
    :param file_name: The name of the file to read in.
    :return: The data np array.
    """
    df: DataFrame = pd.read_csv(file_name)
    return df


def predictions_to_top_n(predictions: List[Prediction], n: int = 5) -> Dict:
    """
    Turns a list of Predictions into a dictionary of user to sorted movie
    ratings.
    :param predictions: The list of Predictions.
    :param n: The number of movies to put in the top n.
    :return: The top n predictions by user.
    """
    prediction_dict: Dict = defaultdict(list)
    for prediction in predictions:  # type: Prediction
        user: str = str(prediction[0])
        movie: str = prediction[1]
        est: np.float64 = prediction[3]
        prediction_dict[user].append((movie, est))

    for user, ratings in prediction_dict.items():  # type: str, List
        ratings.sort(key=lambda x: x[1], reverse=True)
        prediction_dict[user] = ratings[:n]

    return prediction_dict


def test_list_to_dict(test_set: List) -> Dict:
    """
    Transforms the list of test list tuples to a dictionary of user to movie
    ratings.
    :param test_set: The test set to transform.
    :return: The transformed dictionary.
    """
    actual_dict: Dict = defaultdict(dict)
    for user, movie, rating in test_set:  # type: str, str, float
        actual_dict[str(user)][movie] = rating

    return actual_dict


if __name__ == '__main__':
    main()
