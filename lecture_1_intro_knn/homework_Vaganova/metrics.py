import numpy as np


def binary_classification_metrics(y_true, y_pred):
    """
    Computes metrics for binary classification
    Arguments:
    y_true, np array (num_samples) - true labels
    y_pred, np array (num_samples) - model predictions
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # calculate confusion matrix values
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for idx in range(len(y_pred)):
        if y_pred[idx] == 1 and y_true[idx] == 1:
            tp += 1
        elif y_pred[idx] == 1 and y_true[idx] == 0:
            fp += 1
        elif y_pred[idx] == 0 and y_true[idx] == 0:
            tn += 1
        elif y_pred[idx] == 0 and y_true[idx] == 1:
            fn += 1

    # check if division by zero (metric will be set to 0)
    if tp + fp != 0 and tp + fn != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / ((1 / recall) + (1 / precision))

    elif tp + fp == 0 and tp + fn != 0:
        precision = 0
        recall = tp / (tp + fn)
        f1 = 0

    elif tp + fn == 0 and tp + fp != 0:
        precision = tp / (tp + fp)
        recall = 0
        f1 = 0
    else:
        precision = 0
        recall = 0
        f1 = 0

    accuracy = (tp + tn) / (tp + tn + fn + fp)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_true, y_pred):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_true, np array of int (num_samples) - true labels
    y_pred, np array of int (num_samples) - model predictions
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    tp = (y_true == y_pred).sum()

    n_total = len(y_pred)

    accuracy = tp / n_total

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = ((y_true - y_pred) ** 2).sum() / len(y_true)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = ((abs(y_pred - y_true)).sum() / len(y_true))

    return mae
