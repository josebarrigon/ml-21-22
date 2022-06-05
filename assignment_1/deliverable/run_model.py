import joblib
import numpy as np


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x, y = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the trained model
    baseline_model_path = './baseline_model.pickle'
    baseline_model = load_model(baseline_model_path)

    # Predict on the given samples
    y_pred = baseline_model.predict(x)

    ######### LINEAR REGRESSION MODEL ############
    lr_model_path = './linear_regression_model.pickle'
    lr_model = load_model(lr_model_path)

    X_ = lr_model[0] #The expanded matrix
    thetas = lr_model[1]

    y_pred_lr = X_.dot(thetas)

    mse_lr = evaluate_predictions(y_pred_lr, y)
    print('Linear regression model MSE: {}'.format(mse_lr))

    ######### END OF LINEAR REGRESSION MODEL ###########

    ########### Custom Model ##############
    MLP_model_path = './nonlinear_model.pickle'
    MLP_model = load_model(MLP_model_path)

    y_pred_MLP = MLP_model.predict(x)

    mse_MLP = evaluate_predictions(y_pred_MLP, y)
    print('Custom model MSE: {}'.format(mse_MLP))

    ########### END OF Custom Model ##############

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y)
    print('MSE: {}'.format(mse))
