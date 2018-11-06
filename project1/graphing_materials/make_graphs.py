import pandas as pd
import matplotlib.pyplot as plt


def main():
    titanic_accuracies = 'reg_train_test_acc_by_weight_titanic.csv'
    titanic_rand_weight = 'reg_rand_weight_by_weight_titanic.csv'
    titanic_output_name = 'Titanic'
    mnist_accuracies = 'reg_train_test_acc_by_weight_mnist.csv'
    mnist_output_name = 'MNIST_Zoomed'
    mnist_rand_weight = 'reg_rand_weight_by_weight_mnist.csv'


def create_graphic_from_accuracies(input_name, output_name):
    df = pd.read_csv(input_name)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.index[4:7])
    print(df)
    df.columns = ['Regularizer_weight',
                  'Training_error',
                  'Testing_error']
    df.plot.scatter(x='Regularizer_weight', y='Training_error').set_title(
        output_name)
    #plt.show()
    plt.savefig('TrainingErrorReg' + output_name)
    df.plot.scatter(x='Regularizer_weight', y='Testing_error').set_title(
        output_name)
    #plt.show()
    plt.savefig('TestErrorReg' + output_name)


def create_graphs_from_rand_weight(input_name, output_name):
    print(input_name)
    df = pd.read_csv(input_name)
    df = df.drop(df.columns[0], axis=1)
    #df = df.drop(df.index[4:7])
    df.columns = ['Regularizer_weight', 'Rand_weight']
    df.plot.scatter(x='Regularizer_weight', y='Rand_weight').set_title(
        output_name)
    print(df)
    plt.savefig('RandWeight' + output_name)


if __name__ == '__main__':
    main()
