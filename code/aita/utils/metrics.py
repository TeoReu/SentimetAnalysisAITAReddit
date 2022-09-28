import torch
import numpy as np

from sklearn.metrics import precision_score, \
    recall_score, f1_score, matthews_corrcoef, \
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def calculate_metrics(y_true, y_pred):
    """
    Function that return Metrics values for preccision, recall, f1, and matthews
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: metrics
    """
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=1)
    recall = recall_score(y_true=y_true, y_pred=y_pred, zero_division=1)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=1)
    matthews = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    acc = accuracy_score(y_true, y_pred)

    return np.array([acc, precision, recall, f1, matthews])


class Metrics:
    """
    class used to update metrics for training
    """
    def __init__(self):
        self.results = np.zeros(5, dtype=np.float32)
        self.update_counter = 0

    def update_metrics(self, y_true, y_pred):
        new_results = calculate_metrics(y_true=y_true, y_pred=y_pred)
        self.results += new_results
        self.update_counter += 1

    def reset_metrics(self):
        self.results = np.zeros(5, dtype=np.float32)
        self.update_counter = 0

    def calculate_metrics(self):
        metrics = self.results / self.update_counter
        return {"accuracy": metrics[0], "precision": metrics[1], "recall": metrics[2], "f1": metrics[3],
                "MCC": metrics[4]}


def report_metrics(y_true, y_pred) -> dict:
    """
    Function that return Metrics values for preccision, recall, f1, and matthews
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: dictionary containing the metrics
    """
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    matthews = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "MCC": matthews}


def flat_accuracy(preds, labels):
    """
    Flat accuracy for models that return two values, on which we need to apply argmax
    :param preds:
    :param labels:
    :return: the average
    """
    pred_flat = np.argmax(preds, axis=1).flatten().astype(np.float32)
    labels_flat = labels[:, 1].flatten().astype(np.float32)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def generate_confusion_matrix(y_true, y_pred):
    """
    Generates confusion matrix based on true values and predictions
    :param y_true:
    :param y_pred:
    :return:
    """
    return confusion_matrix(y_true, y_pred)
