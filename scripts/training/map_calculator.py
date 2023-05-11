"""
This module contains a function for calculating the Mean Average Precision (mAP)
for a given dataframe and ground truth data.
"""


import numpy as np
import pandas as pd


def calculate_map(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    """
    Calculate the mean Average Precision (mAP) for a given dataframe and ground truth data.

    Args:
        predictions (pd.DataFrame): The dataframe containing prediction data.
        ground_truth (pd.DataFrame): The dataframe containing ground truth data.

    Returns:
        float: The mean Average Precision (mAP) value.
    """
    user_aps = []

    for user_id in ground_truth.index:
        user_df = predictions[predictions["userid"] == user_id]\
            .sort_values(by="prediction", ascending=False)
        num_correct = 0
        precision = []
        recall = []

        for i, movie_id in enumerate(user_df["movieid"]):
            if movie_id in ground_truth.loc[user_id]["movieid"]:
                num_correct += 1
            precision.append(num_correct / (i + 1))
            recall.append(num_correct / len(ground_truth.loc[user_id]["movieid"]))

        # Calculate average precision
        average_precision = 0
        for i, (prec, rec) in enumerate(zip(precision, recall)):
            if i == 0 or rec != recall[i - 1]:
                average_precision += prec * (rec - recall[i - 1] if i > 0 else rec)

        user_ap = average_precision / len(ground_truth.loc[user_id]["movieid"])
        user_aps.append(user_ap)

    return np.mean(user_aps)
