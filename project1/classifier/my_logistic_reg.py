import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from typing import Tuple, Dict, List


class MyLogisticReg:
    """
    A class for Linear Logistic Regression Models.
    """
    def __init__(self,
                 learning_rate: float = None,
                 regularizer_weight: float = .5,
                 sgd_flag: bool = False,
                 log_flag: bool = True,
                 batch_size: int = 10,
                 save_flag: bool = False,
                 max_iterations: int = None,
                 mnist_flag: bool = False) -> None:
        """
        Initializes the training parameters. weights are not initialized here
        as we do not know the number of features yet.
        """
        self._regularizer_weight: float = regularizer_weight
        self._sgd_flag: bool = sgd_flag
        self._log_flag: bool = log_flag
        self._f_weights: np.array = None
        self._i_weight: np.float64 = None
        self._batch_size: int = batch_size
        self._save_flag: int = save_flag
        self._max_iterations: int = max_iterations
        self._mnist_flag: bool = mnist_flag
        if learning_rate is not None:
            self._learning_rate = learning_rate
        else:
            if mnist_flag:
                self._learning_rate: float = .001
            else:
                self._learning_rate: float = .00001
        if max_iterations is not None:
            self._max_iterations = max_iterations
        else:
            if mnist_flag:
                self._max_iterations = 50000
            else:
                self._max_iterations = 100000

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits a model to data by finding values of the weight array that
        minimize logistic loss.
        :param X: The matrix containing the training instances.
        :param y: The array containing the training labels.
        :return: None. This function produces a trained weight array as a side
        effect.
        """
        tf.reset_default_graph()

        # numpy pre-processing:

        # If running MNIST don't forget to scale.
        if self._mnist_flag:
            X = np.divide(X, 255.0)
            y[y == 8] = 0
            y[y == 9] = 1

        # Grab the shapes needed for the tf placeholders.
        x_shape: Tuple[int, int] = X.shape
        y_shape: Tuple[int] = y.shape
        if self._sgd_flag:
            x_shape = (self._batch_size, x_shape[1])
            y_shape = (self._batch_size,)


        # Set up TensorFlow graph:

        # Boiler-plate initialization of variables:

        f_weights: tf.Variable
        i_weight: tf.Variable
        features: tf.Tensor
        labels: tf.Tensor
        learning_rate: tf.Tensor
        regularizer_weight: tf.Tensor

        f_weights, \
        i_weight, \
        features, \
        labels, \
        learning_rate, \
        regularizer_weight, = self.__fit_setup(x_shape, y_shape)

        # Perform training

        assign_f_weights: tf.Tensor
        assign_i_weight: tf.Tensor
        assign_f_weights, assign_i_weight = self.__fit_train(
            f_weights,
            i_weight,
            features,
            labels,
            learning_rate,
            regularizer_weight)

        # Set up logging

        if self._log_flag:
            merged_summaries: tf.Tensor = self.__log()

        # If stochastic gradient descent is being used then we will need
        # to shuffle the data set.
        if self._sgd_flag:
            x_and_y: np.array = np.append(X, y.reshape(y.shape[0], 1), axis=1)
            np.random.shuffle(x_and_y)

        # Construct a session with the config equal to the following if not
        # running MNIST
        # config=tf.ConfigProto(device_count={'GPU': 0})
        # Starting the actual session
        with tf.Session() as sess:  # type: tf.Session
            # Creates our file_writer used for logging useful metrics
            if self._log_flag:
                file_writer: tf.summary.FileWriter = tf.summary.FileWriter(
                    '.', sess.graph)

            # Initialize the variables in fit set up
            sess.run(tf.global_variables_initializer())

            # Keep the weights from 100 iterations behind in order to measure
            # the difference as described in the project spec
            last_f_weights: np.array = None
            last_i_weight: np.float64 = None

            # Keep track of the number of iterations
            num_iterations: int = 0
            # Start the sgd indexes at -batch_size from where they should be
            # so we can keep all of the sgd stuff inside one if statement
            # in the main loop
            start_index: int = -self._batch_size
            end_index: int = 0
            # Assign the portion of the data set we are going to be working
            # with, in this case all of it.
            current_x: np.array = X
            current_y: np.array = y
            while num_iterations < self._max_iterations:
                # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # If we are using stochastic gradient descent
                if self._sgd_flag:
                    # If our index is outside of the bounds we need to
                    # shuffle and reset
                    if end_index + self._batch_size >= x_and_y.shape[0]:
                        np.random.shuffle(x_and_y)
                        start_index = -self._batch_size
                        end_index = 0
                    # Increment the indexes every iteration
                    start_index += self._batch_size
                    end_index += self._batch_size
                    # Grab the ten instances we will use for the sgd
                    current_x_and_y: np.array = x_and_y[start_index:end_index]
                    # Split them into x and y
                    current_x: np.array = current_x_and_y[:, 0:-1]
                    current_y: np.array = current_x_and_y[:, -1]
                # What will be fed into the model
                feeder: Dict = {
                        features: current_x,
                        labels: current_y}

                # Grab the current weights from tf
                current_f_weights: np.array
                current_i_weight: np.array
                current_f_weights, current_i_weight = sess.run(
                    [assign_f_weights, assign_i_weight], feed_dict=feeder)
                if num_iterations % 100 == 0:
                    # If logging is enabled then loss is logged every 100
                    # iterations.
                    if self._log_flag:
                        self.__run_summary(
                            sess,
                            merged_summaries,
                            file_writer,
                            feeder,
                            num_iterations)

                    # Check to see if we've reached our stop condition
                    if last_f_weights is not None:
                        diff: float = self.__get_diff(current_i_weight,
                                                      last_i_weight,
                                                      current_f_weights,
                                                      last_f_weights,
                                                      x_shape[1])
                        # We are done
                        if diff < pow(10, -4):
                            break
                    # Keep current weights for next diff check
                    last_i_weight = current_i_weight
                    last_f_weights = current_f_weights
                num_iterations += 1

        # Grab our new weights for predictions
        self._f_weights = current_f_weights
        self._i_weight = current_i_weight

    def predict(self, X: np.array) -> np.array:
        """
        Makes predictions given data
        :param X: The data to make predictions for using the weights.
        :return: The predictions.
        """
        if self._sgd_flag:
            X = np.divide(X, 255.0)
        # Turn the X np.array into a tensor
        features: tf.Tensor = tf.placeholder(
            tf.float64, X.shape, name="features")

        # Create a weight for each feature
        f_weights: tf.Tensor = tf.placeholder(
            tf.float64, (X.shape[1]), name="f_weights")

        # Create a weight for the intercept (w_0)
        i_weight: tf.Tensor = tf.placeholder(
            tf.float64,  (), name="i_weight")

        # Make some predictions
        predictions: tf.Tensor = self.__make_predictions(
            features, f_weights, i_weight)

        with tf.Session() as sess:
            actual_preds: np.array = sess.run(
                predictions, feed_dict={
                    features: X,
                    f_weights: self._f_weights,
                    i_weight: self._i_weight})

        # Remember to actually turn the output into real predictions
        actual_preds[actual_preds > 0] = 1
        actual_preds[actual_preds <= 0] = 0
        return actual_preds

    def evaluate(self, y_test: np.array, y_pred: np.array) -> float:
        """
        Evaluated a classifier by reporting the error rate given the test
        labels and test predictions.
        :param y_test: The test labels.
        :param y_pred: The test predictions.
        :return: The error rate
        """
        # Remember to change the labels to 0 and 1 for MNIST
        if self._mnist_flag:
            y_test[y_test == 8] = 0
            y_test[y_test == 9] = 1
        error_rate: float = (np.sum(
            np.equal(y_test, y_pred).astype(np.float)) / y_test.size)
        return 1 - error_rate

    def __get_diff(self,
                   current_i_weight: float,
                   last_i_weight: float,
                   current_f_weights: np.array,
                   last_f_weights: np.array,
                   num_features: int):
        """
        Calculates the difference (as defined in the assignment) between a pair
         of weights.
        :param current_i_weight: The current intercept weight.
        :param last_i_weight: The last intercept weight.
        :param current_f_weights: The current feature weights.
        :param last_f_weights: The last feature weights.
        :param num_features: The number of features.
        :return: The difference between the weights as defined in the
        assignment.
        """
        return (np.abs(current_i_weight - last_i_weight) +
                np.sum(np.abs(current_f_weights - last_f_weights))) / \
               (num_features + 1)


    def __run_summary(self,
                      sess: tf.Session,
                      merged_summaries: tf.Tensor,
                      file_writer: tf.summary.FileWriter,
                      feeder: Dict,
                      num_iterations: int) -> None:
        """
        Creates and writes a TF summary.
        :param sess: The session to run this on
        :param merged_summaries: The summaries to write.
        :param file_writer: The writer to use.
        :param feeder: The dictionary of values to feed into the
        merged_summaries op.
        :param num_iterations: The number of iterations.
        :return: Nothing, writes summaries as a side effect.
        """
        current_summary = sess.run(
            merged_summaries,
            feed_dict=feeder)
        file_writer.add_summary(current_summary, num_iterations)

    def __fit_setup(self, x_shape: Tuple, y_shape: Tuple) -> Tuple[
            tf.Variable,
            tf.Variable,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor]:
        """
        Performs the setup portion of the fit functions TF graph. This consists
        of initializing variables, and populating Tensors.
        :param x_shape: A tuple containing the shape of the training instances
        array.
        :param y_shape: A tuple containing the shape of the training labels
        array.
        :return: A tuple containing the variables f_weights, and i_weight. As
        well as the tensors features, labels, learning_rate, and
        regularizer_weight.
        """

        # General set up / initialization
        with tf.name_scope("fit_setup"):
            # Variable set up
            with tf.name_scope("fit_variables_setup"):
                # Create a weight for each feature
                # They should be small random numbers
                f_weights: tf.Variable = tf.get_variable(
                                          "f_weights",
                                          (x_shape[1],),
                                          dtype=tf.float64,
                                          initializer=
                                          tf.initializers.random_uniform(
                                              minval=-1,
                                              maxval=1))

                # Create a weight for the intercept (w_0)
                i_weight: tf.Variable = tf.get_variable(
                    "i_weight",
                    (),
                    dtype=tf.float64,
                    initializer=tf.initializers.random_uniform(
                        minval=-1, maxval=1))

            # Non variable tensor set up
            with tf.name_scope("fit_tensor_setup"):
                # Turn the X np.array into a tensor
                features: tf.Tensor = tf.placeholder(
                    tf.float64, x_shape, name="features")

                # Turn the y np.array into a tensor
                labels: tf.Tensor = tf.placeholder(
                    tf.float64, (y_shape[0],), name="labels")

                # Grab the learning rate for this iteration
                learning_rate: tf.Tensor = tf.constant(
                    self._learning_rate,
                    dtype=tf.float64,
                    name="learning_rate")

                # Grab the regularizer_weight
                regularizer_weight: tf.Tensor = tf.constant(
                    self._regularizer_weight,
                    dtype=tf.float64,
                    name="regularizer_weight")

        return (f_weights,
                i_weight,
                features,
                labels,
                learning_rate,
                regularizer_weight)

    def __fit_train(self,
                  f_weights: tf.Variable,
                  i_weight: tf.Variable,
                  features: tf.Tensor,
                  labels: tf.Tensor,
                  learning_rate: tf.Tensor,
                  regularizer_weight: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """
        Creates the TF graph that performs the actual training.
        :param f_weights: The tensor of weights for each feature. (To be
        trained).
        :param i_weight: The weight for the intercept. (To be trained).
        :param features: The tensor of features. (Used in conjunction with the
        weights to predict labels).
        :param labels: The tensor of labels. (Used to quantify the loss of each
        prediction).
        :param learning_rate: The learning rate. In this case, what fraction
        of the gradient to subtract from the weight.
        :param regularizer_weight: The regularizer weight. Determines how large
        the penalty for large feature weights is.
        :return: The final TF operation that will assign new weights to
        f_weights and i_weight.
        """
        # Actual calculations begin:
        with tf.name_scope("fit_training"):

            predicted_labels: tf.Tensor = \
                self.__make_predictions(features, f_weights, i_weight)

            with tf.name_scope("negative_cost"):
                # Multiply the actual labels by the predicted labels.
                actual_mul_predict: tf.Tensor = tf.multiply(
                    labels, predicted_labels, name="actual_mul_predict")

                # log(exp(predicted_labels) + 1) with some extra special tf
                # spice
                soft_plus_predict: tf.Tensor = tf.nn.softplus(
                    predicted_labels, name="soft_plus_predict")

                # Subtract actual_mul_predict by soft_plus_predict
                actual_mul_predict_minus_soft_plus_predict: tf.Tensor = \
                    tf.subtract(
                        actual_mul_predict,
                        soft_plus_predict,
                        name="actual_mul_predict_minus_sp_predict")

                # Reduce sum the actual_mul_predict_mins_soft_plus_predict to
                # get the cost before regularization
                cost_without_regularizer: tf.Tensor = tf.reduce_sum(
                    actual_mul_predict_minus_soft_plus_predict,
                    name="cost_without_regularizer")

            with tf.name_scope("regularization"):
                # Make our regularizer
                # It is important to remember not to add the intercept to this.
                # There is not reason to encourage a small intercept
                regularizer: tf.Tensor = tf.tensordot(
                    f_weights, f_weights, [0, 0], name="regularizer")

                # Multiply the regularizer by its weight
                # tf.identity is apparently the best way to rename a tensor
                # that is returned from an operation that does not allow the
                # name option
                weighted_regularizer: tf.Tensor = tf.identity(
                    tf.scalar_mul(regularizer_weight, regularizer),
                    name="weighted_regularizer")

            with tf.name_scope("final_cost"):
                # Final cost is the regularizer multiplied by its weight minus
                # the cost before regularization.
                final_cost: tf.Tensor = tf.subtract(
                    weighted_regularizer,
                    cost_without_regularizer,
                    name="final_cost")

                tf.summary.scalar('final_cost', final_cost)

            with tf.name_scope("gradients"):
                # Get the gradient of our final_cost with respect to the
                # feature weights and intercept weight
                f_weights_grad: tf.Tensor
                i_weight_grad: tf.Tensor
                f_weights_grad, i_weight_grad = tf.gradients(
                    final_cost, [f_weights, i_weight], name="gradient_list")

                tf.summary.histogram("f_weights_grad", f_weights_grad)
                tf.summary.histogram("i_weight_grad", i_weight_grad)

                # Feature weight grad times learning rate
                learning_rate_mul_f_weight_grad: tf.Tensor = tf.identity(
                    tf.scalar_mul(learning_rate, f_weights_grad),
                    name="learning_rate_mul_f_weight_grad")

                # Intercept weight grad times learning rate
                learning_rate_mul_i_weight_grad: tf.Tensor = tf.identity(
                    tf.scalar_mul(learning_rate, i_weight_grad),
                    name="learning_rate_mul_i_weight_grad")

            with tf.name_scope("update_weights"):
                # Calculate and assign new feature weights by subtracting our
                # final value from the feature weights of the last iteration
                new_f_weights: tf.Tensor = tf.subtract(
                    f_weights,
                    learning_rate_mul_f_weight_grad,
                    name="new_f_weights")

                # Calculate and assign the new intercept weight by subtracting
                # our final value from the intercept weight of the last
                # iteration
                new_i_weight: tf.Tensor = tf.subtract(
                    i_weight,
                    learning_rate_mul_i_weight_grad,
                    name="new_i_weights")

                tf.summary.histogram("new_f_weights_hist", new_f_weights)
                tf.summary.histogram("new_i_weight_hist", new_i_weight)

                # Now assign our new weights to our feature weights variable
                assign_f_weights: tf.Tensor = tf.assign(
                    f_weights, new_f_weights, name="assign_f_weights")
                assign_i_weight = tf.assign(
                    i_weight, new_i_weight, name="assign_i_weight")

        return assign_f_weights, assign_i_weight

    def __make_predictions(self,
                           features: tf.Tensor,
                           f_weights: tf.Tensor,
                           i_weight: tf.Tensor) -> tf.Tensor:
        """
        Makes predictions for a vector of instances given feature weights
        and the intercept weight.
        :param features: The features whose labels should be predicted.
        :param f_weights: The weights corresponding to each feature.
        :param i_weight: The intercept weight.
        :return: A tensor of predicted labels.
        """
        # Make predictions
        with tf.name_scope("prediction"):
            # Multiply the features by their weights
            features_mul_their_weights: tf.Tensor = tf.tensordot(
                features, f_weights, [1, 0],
                name="features_mul_their_weights")

            # Add the intercept to create the final prediction
            predicted_labels: tf.Tensor = tf.add(
                features_mul_their_weights,
                i_weight,
                name="predicted_labels")
        return predicted_labels

    def __log(self) -> tf.Tensor:
        with tf.name_scope("logging"):
            # Add saver to ops graph
            # saver = tf.train.Saver()
            return tf.summary.merge_all()


def main() -> None:
    """
    Perform all required testing here.
    :return: Nothing.
    """
    mnist_file_name = 'mnist-train.csv'
    titanic_file_name = 'titanic_train.csv'
    # full_gradient_versus_stochastic()
    # effects_of_regularizers(mnist_file_name, True)
    # effects_of_regularizers(titanic_file_name, False)
    # cross_validation(mnist_file_name, True)
    # cross_validation(titanic_file_name, False)
    # full_data_train(mnist_file_name, 'mnist_classifier.pkl', True)
    # full_data_train(titanic_file_name, 'titanic_classifier.pkl', False)


def full_gradient_versus_stochastic() -> None:
    """
    Examines convergence speeds of full gradient vs stochastic. Uses tfs built
    in logging/event tracking to build graphs.
    :return: Nothing.
    """
    # Preprocessing
    full_mnist: pd.DataFrame = pd.read_csv('mnist-train.csv')
    labels: pd.DataFrame = full_mnist.iloc[:, 0]
    features: pd.DataFrame = full_mnist.iloc[:, 1:full_mnist.shape[1]]

    # Create object with sgd turned off
    logistic_regression_full: MyLogisticReg = MyLogisticReg(
        sgd_flag=False, mnist_flag=True)
    logistic_regression_full.fit(features.values, labels.values)

    # Create object with sgd turned on
    logistic_regression_full = MyLogisticReg(
        sgd_flag=True, mnist_flag=True)
    logistic_regression_full.fit(features.values, labels.values)


def effects_of_regularizers(file_name: str, mnist_flag) -> None:
    """
    Tests the effects of a number of regularizer weights on classifier
    performance, and weight of the random feature.
    :param file_name: The file name of the data set to check.
    :param mnist_flag: Whether or not we are using the mnist dataset.
    :return:
    """
    # Regularizer weights / 2 because I don't divide by two in the graph.
    regularizer_weights: List[float] = [0, 0.005, 0.05, .5, 5, 50, 500]

    # Test
    full_data: pd.DataFrame = pd.read_csv(file_name)
    num_features: int = full_data.shape[1]
    split_full_data: List[pd.DataFrame] = split(full_data, mnist_flag)
    train_data: pd.DataFrame = pd.concat(split_full_data[0:7])
    test_data: pd.DataFrame = pd.concat(split_full_data[7:10])
    training_x: pd.DataFrame = train_data.iloc[:, 1:num_features]
    training_y: pd.DataFrame = train_data.iloc[:, 0]
    testing_x: pd.DataFrame = test_data.iloc[:, 1:num_features]
    testing_y: pd.DataFrame = test_data.iloc[:, 0]

    reg_weight_train_test_accuracy: List[List[float]] = []
    reg_weight_rand_weight: List[List[float]] = []
    for reg_weight in regularizer_weights:
        logistic_regression: MyLogisticReg = \
            MyLogisticReg(regularizer_weight=reg_weight, mnist_flag=mnist_flag)
        logistic_regression.fit(training_x.values, training_y.values)
        training_predict: np.array = logistic_regression.predict(
            training_x.values)
        testing_predict: np.array = logistic_regression.predict(
            testing_x.values)
        training_accuracy: np.array = logistic_regression.evaluate(
            training_y, training_predict)
        testing_accuracy: np.array = logistic_regression.evaluate(
            testing_y, testing_predict)
        reg_weight_train_test_accuracy.append(
            [reg_weight, training_accuracy, testing_accuracy])
        reg_weight_rand_weight.append(
            [reg_weight, np.abs(logistic_regression._f_weights[-1])])

    # Save these values to a csv for later processing
    reg_weight_train_test_accuracy_df: pd.DataFrame = pd.DataFrame(
        reg_weight_train_test_accuracy)
    reg_weight_train_test_accuracy_df.to_csv(
        "reg_train_test_acc_by_weight_" + file_name)
    reg_weight_rand_weight_df: pd.DataFrame = pd.DataFrame(
        reg_weight_rand_weight)
    reg_weight_rand_weight_df.to_csv(
        "reg_rand_weight_by_weight_" + file_name)


def cross_validation(file_name: str, mnist_flag: bool) -> None:
    """
    Performs cross validation using the given data set.
    :param file_name: The data set to use.
    :param mnist_flag: Whether or not we are using the mnist dataset.
    :return: Nothing.
    """
    full_data: pd.DataFrame = pd.read_csv(file_name)
    # Shuffle the dataframe
    full_data = full_data.sample(frac=1)
    num_features = full_data.shape[1]
    split_full = split(full_data, mnist_flag)
    accuracies: List = []
    for i in range(10):
        logistic_regression_full: MyLogisticReg = MyLogisticReg(
            mnist_flag=mnist_flag)
        test_data = pd.concat(split_full[i:i+1])
        train_data = pd.concat(
            [item for index, item in
             enumerate(split_full) if index != i])
        training_x = train_data.iloc[:, 1:num_features]
        training_y = train_data.iloc[:, 0]
        testing_x = test_data.iloc[:, 1:num_features]
        testing_y = test_data.iloc[:, 0]
        logistic_regression_full.fit(training_x, training_y)
        testing_predict = logistic_regression_full.predict(testing_x.values)
        testing_accuracy = logistic_regression_full.evaluate(
            testing_y, testing_predict)
        accuracies.append(testing_accuracy)

    accuracies_df = pd.DataFrame(accuracies)
    accuracies_df.to_csv("accuracies_cross_val.csv")


def full_data_train(
        file_name: str,
        output_name: str,
        mnist_flag: bool) -> None:
    max_iters: int = 200000
    if mnist_flag:
        max_iters = 100000
    full_data: pd.DataFrame = pd.read_csv(file_name)
    num_features: int = full_data.shape[1]
    training_x: pd.DataFrame = full_data.iloc[:, 1:num_features]
    training_y: pd.DataFrame = full_data.iloc[:, 0]
    logistic_regression_full: MyLogisticReg = MyLogisticReg(
        mnist_flag=mnist_flag, max_iterations=max_iters)
    logistic_regression_full.fit(training_x.values, training_y.values)
    pickle.dump(logistic_regression_full, open(output_name, "wb"))


def split(df: pd.DataFrame, mnist_flag: bool) -> List[pd.DataFrame]:
    """
    Splits the data frame into 10 roughly equal partitions keeping the ratio
    of two classes the same.
    :param df: The data frame to split.
    :param mnist_flag: Whether or not it is MNIST.
    :return: A list containing each of the 10 partitions.
    """
    folded = []
    if mnist_flag:
        nines: pd.DataFrame = df[df.iloc[:, 0] == 9]
        eights: pd.DataFrame = df[df.iloc[:, 0] == 8]
        nines_list: List[pd.DataFrame] = np.array_split(nines, 10)
        eights_list: List[pd.DataFrame] = np.array_split(eights, 10)
        for nine, eight in zip(nines_list, eights_list):
            folded.append(pd.concat([nine, eight]))
    else:
        surviveds: pd.DataFrame = df[df.iloc[:, 0] == 1]
        deads: pd.DataFrame = df[df.iloc[:, 0] == 0]
        survived_list: List[pd.DataFrame] = np.array_split(surviveds, 10)
        dead_list: List[pd.DataFrame] = np.array_split(deads, 10)
        for survived, dead in zip(survived_list, dead_list):
            folded.append(pd.concat([survived, dead]))
    return folded


if __name__ == '__main__':
    main()
