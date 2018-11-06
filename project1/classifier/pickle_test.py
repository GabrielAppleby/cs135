import pickle
import numpy as np
import pandas as pd
from my_logistic_reg import MyLogisticReg


def titanic():
    full_data: pd.DataFrame = pd.read_csv('titanic_train.csv')
    num_features: int = full_data.shape[1]
    training_x: pd.DataFrame = full_data.iloc[:, 1:num_features]
    training_y: pd.DataFrame = full_data.iloc[:, 0]
    classifier: MyLogisticReg = pickle.load(
        open('titanic_classifier.pkl', 'rb'))
    predictions: np.array = classifier.predict(training_x)
    print(classifier.evaluate(training_y, predictions))


def mnist():
    full_data: pd.DataFrame = pd.read_csv('mnist-train.csv')
    num_features: int = full_data.shape[1]
    training_x: pd.DataFrame = full_data.iloc[:, 1:num_features]
    training_y: pd.DataFrame = full_data.iloc[:, 0]
    classifier: MyLogisticReg = pickle.load(
        open('mnist_classifier.pkl', 'rb'))
    predictions: np.array = classifier.predict(training_x)
    print(classifier.evaluate(training_y, predictions))


if __name__ == '__main__':
    titanic()
    mnist()
