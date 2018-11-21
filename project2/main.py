import _io
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from text_preprocessor import TextPreprocessor
from typing import Tuple, Dict, List

TITLE: str = 'TITLE'
X_LABEL: str = 'xlabel'
C: str = 'C'
NEIGHBORS: str = 'n_neighbors'
MEAN_TEST_SCORE: str = 'mean_test_score'
MEAN_TRAIN_SCORE: str = 'mean_train_score'
STD_TEST_SCORE: str = 'std_test_score'
PARAMS: str = 'params'
BAG_OF_WORDS: str = ' tfidf'
GLOVE: str = ' glove'

GLOVE_FILE: str = 'resources/glove.6B.50d.txt'
GLOVE_DIMENSION: int = 50
TRAIN_FILE: str = 'resources/trainreviews.txt'
TEST_FILE: str = 'resources/testreviewsunlabeled.txt'
TRAIN_SEPARATOR: str = '\t'
TRAIN_COLUMNS: List = ['Text', 'Sentiment']

TRANSFORM_PARAMS: Dict = {'tfidftransformer__use_idf': [True, False]}

KNN_PARAMS: Dict = {NEIGHBORS: [1, 5, 10, 15, 20, 25, 30, 35, 40, 100],
                    'weights': ['uniform'],
                    'p': [2]}

KNN_GRAPH_LABELS: Dict = {TITLE: 'K Nearest Neighbors Classification',
                          X_LABEL: NEIGHBORS}

LOGISTIC_GRAPH_LABELS: Dict = {TITLE: 'Logistic Classification', X_LABEL: C}

LOGISTIC_PARAMS: Dict = {C: [.01, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                         'solver': ['liblinear'],
                         'penalty': ['l2'],
                         'max_iter': [100]}

SVC_PARAMS: Dict = {C: [.01, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                    'kernel': ['linear'],
                    'gamma': ['scale']}

SVC_GRAPH_LABELS: Dict = {TITLE: 'C-Support Vector Classification',
                          X_LABEL: C}

CLASSIFIERS: List = [(KNeighborsClassifier(), KNN_PARAMS, KNN_GRAPH_LABELS),
                     (LogisticRegression(),
                      LOGISTIC_PARAMS,
                      LOGISTIC_GRAPH_LABELS),
                     (SVC(), SVC_PARAMS, SVC_GRAPH_LABELS)]


def main() -> None:
    """
    Gets the confidence of the best classifier, and produces graphs of some
    of the tests on all of the classifiers and data representations.
    :return: Nothing
    """
    labeled_data: np.array
    target: np.array
    unlabeled_data: np.array
    labeled_data, target = read_data(TRAIN_FILE)
    unlabeled_data, _ = read_data(TEST_FILE)

    # Why all the copies Gabe? Well, originally I was only testing so I decided
    # to modify the data array in place within many parts of my code. Now that
    # others need to use my code it makes more sense just to copy it, as
    # otherwise the code would only work in a specific order. Which would
    # not allow much testing.

    # best_classifier_ci(np.copy(labeled_data), target)
    # test_classifiers_and_data_reps(np.copy(labeled_data), target)

    # Lets make some predictions.
    logistic_pipeline: Pipeline = make_pipeline(
        CountVectorizer(), TfidfTransformer(), LogisticRegression(
            C=1.5, solver='liblinear'))
    logistic_pipeline.fit(np.copy(labeled_data), target)
    predictions: np.array = logistic_pipeline.predict(unlabeled_data)
    integer_predictions: np.array = predictions.astype(int, copy=True)
    np.savetxt('predicted-labels.txt', integer_predictions, fmt='%i')


def read_data(file_name: str) -> Tuple[np.array, np.array]:
    """
    Reads in the data from the train reviews.
    :param file_name: The name of the file to read in.
    :return: The data and the target np arrays.
    """
    train_data: pd.DataFrame = pd.read_csv(
        file_name, sep=TRAIN_SEPARATOR, names=TRAIN_COLUMNS)
    data: np.array = train_data[TRAIN_COLUMNS[0]].values
    target: np.array = train_data[TRAIN_COLUMNS[1]].values
    return data, target


def best_classifier_ci(data: np.array, target: np.array) -> None:
    """
    Produces the confidence interval for the best classifier.
    :param data: The data to use to perform cross validation
    :param target: The labels of the data to use for cross validation.
    :return: Nothing.
    """
    count_vectorizer: CountVectorizer = CountVectorizer()
    count_data: np.array = count_vectorizer.fit_transform(data)
    tfidf_transformer: TfidfTransformer = TfidfTransformer()
    tfidf_data: np.array = tfidf_transformer.fit_transform(count_data)
    clf = LogisticRegression(C=1.5, solver='liblinear')
    scores: np.array = cross_val_score(clf, tfidf_data, target, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def test_classifiers_and_data_reps(data, target) -> None:
    """
    Tests different classifier configurations, and data representations.
    :param data: The data to use to test these with.
    :param target: The labels of the data.
    :return: Nothing.
    """
    bag_of_words(data, target)
    glove(data, target)


def bag_of_words(data: np.array, target: np.array) -> None:
    """
    Tests the tfidf representation on all of the classifiers and
    configurations.
    :param data: The data to use for the testing.
    :param target: The datas labels.
    :return: Nothing.
    """
    count_vectorizer: CountVectorizer = CountVectorizer(
        tokenizer=TextPreprocessor().process)
    count_data: np.array = count_vectorizer.fit_transform(data)
    tfidf_transformer: TfidfTransformer = TfidfTransformer()
    tfidf_data: np.array = tfidf_transformer.fit_transform(count_data)
    graph_param_metrics(tfidf_data, target, BAG_OF_WORDS)


def glove(data: np.array, target: np.array):
    """
    Tests the glove representation on all of the classifiers and
    configurations.
    :param data: The data to use for testing.
    :param target: The datas labels.
    :return: Nothing.
    """
    text_preprocessor: TextPreprocessor = TextPreprocessor()
    data = text_preprocessor.transform(data)
    glove_dict: Dict = glove_csv_to_dict(GLOVE_FILE)
    glove_rep_of_data: np.array = get_glove_rep_of(data, glove_dict)
    graph_param_metrics(glove_rep_of_data, target, GLOVE)


def glove_csv_to_dict(glove_file) -> Dict:
    """
    Reads in the glove file and puts it in a dictionary.
    :param glove_file: The glove file to read in.
    :return: The dictionary mapping words to their glove representations.
    """
    glove_dict: Dict = {}
    with open(glove_file, 'r') as file:  # type: _io.TextIOWrapper
        glove_lines: List = file.read().splitlines()
        for line in glove_lines:  # type: str
            split_glove_line: List[str] = line.split(' ')
            word: str = split_glove_line[0]
            word_glove_values: np.array = np.asarray(
                [float(x) for x in split_glove_line[1:]], dtype=np.float64)
            glove_dict[word] = word_glove_values
    return glove_dict


def get_glove_rep_of(data: np.array, glove_dict) -> np.array:
    """
    Gets the glove representation of the train reviews data.
    :param data: The data to get the glove representation of.
    :param glove_dict: The mapping of words to glove representations.
    :return: The glove representation of the train reviews data.
    """
    glove_rep_of_reviews: np.array = np.zeros((len(data), GLOVE_DIMENSION))
    for counter, review in enumerate(data):
        for word in review:
            glove_rep_of_word: np.array = glove_dict.get(word)
            if glove_rep_of_word is not None:
                glove_rep_of_reviews[counter] += glove_rep_of_word
    return glove_rep_of_reviews


def graph_param_metrics(data, target, name) -> None:
    """
    Performs a grid search of the parameters for each classifier and graphs
    the results.
    :param data: The data the classifiers will use to train.
    :param target: The label of the data the classifiers will use to train.
    :param name: The name of the data representation.
    :return: Nothing
    """
    for clf, params, labels in CLASSIFIERS:  # type: ClassifierMixin, Dict, Dict
        test_means: List
        test_stds: List
        train_means: List
        params: Dict
        test_means, test_stds, train_means, params = grid_search(
            clf, params, data, target)
        graph_helper(test_means, train_means, params, labels, name)
        #print_helper(test_means, test_stds, params)


def graph_helper(
        mean_test_scores: List,
        mean_train_scores: List,
        params: Dict,
        labels: Dict,
        name: str) -> None:
    """
    Creates and saves the parameter graphs.
    :param mean_test_scores: The mean test scores for each parameter value.
    :param mean_train_scores: The mean training scores for each parameter
    value.
    :param params: The parameters and their corresponding values.
    :param labels: The labels for the graph.
    :param name: Name of data representation needed for saving the graph.
    :return: Nothing.
    """
    param_values = \
        [param[labels[X_LABEL]] for param in params]
    plt.figure()
    plt.plot(param_values, mean_test_scores, 'r', label='Test')
    plt.plot(param_values, mean_train_scores, 'b', label='Train')
    plt.ylabel('Error')
    plt.xlabel(labels[X_LABEL])
    plt.title(labels[TITLE] + name)
    plt.legend()
    plt.savefig(labels[TITLE] + name + '.png')


def print_helper(
        mean_test_scores: List,
        test_stds: List,
        params: Dict) -> None:
    """
    Prints the values that would be graphed.
    :param mean_test_scores: The mean test scores for each parameter value.
    :param mean_train_scores: The mean training scores for each parameter
    value.
    :param params: The parameters and their corresponding values.
    :param labels: The labels for the graph.
    :param name: Name of data representation needed for saving the graph.
    :return: Nothing.
    """
    for test_mean, std, params in zip(mean_test_scores, test_stds, params):
        print("Test: %0.3f (+/-%0.03f) for %r"
              % (test_mean, std, params))


def grid_search(clf: ClassifierMixin,
                params_to_search: Dict,
                data: np.array,
                target: np.array) -> Tuple[List, List, List, Dict]:
    """
    Performs a grid search of the given parameters for the given classifier.
    :param clf: The classifier to test.
    :param params_to_search: The different parameters to tet.
    :param data: The data to test on.
    :param target: The labels of the data to test on.
    :return: The mean test scores, and mean training scores for each parameter
    value as well as the parameters and their values.
    """
    gs_clf: GridSearchCV = GridSearchCV(
        clf,
        params_to_search,
        cv=5,
        iid=False,
        n_jobs=-1,
        return_train_score=True,
        error_score='raise')
    gs_clf: GridSearchCV = gs_clf.fit(data, target)
    test_means = np.subtract(1, gs_clf.cv_results_[MEAN_TEST_SCORE])
    test_stds = gs_clf.cv_results_[STD_TEST_SCORE]
    train_means = np.subtract(1, gs_clf.cv_results_[MEAN_TRAIN_SCORE])
    params = gs_clf.cv_results_[PARAMS]
    return test_means, test_stds, train_means, params


if __name__ == '__main__':
    main()
