"""
This module provides functions for calculating Discounted Cumulative Gain (DCG) and
Normalized Discounted Cumulative Gain (NDCG) for evaluating ranking algorithms.
"""

import numpy as np


def dcg_at_k(relevance_scores, k):
    """
    Calculate Discounted Cumulative Gain (DCG) at position k.

    Args:
        relevance_scores (array-like): Array of relevance scores.
        k (int): Position to calculate DCG at.

    Returns:
        float: DCG at position k.
    """
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        return relevance_scores[0] \
               + np.sum(relevance_scores[1:] / np.log2(np.arange(2, relevance_scores.size + 1)))
    return 0.0


def ndcg_at_k(relevance_scores, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at position k.

    Args:
        relevance_scores (array-like): Array of relevance scores.
        k (int): Position to calculate NDCG at.

    Returns:
        float: NDCG at position k.
    """
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(relevance_scores, k) / dcg_max


def calculate_ndcg(data_frame, ground_truth, k):
    """
    Calculate the average NDCG for a given dataframe and ground truth ratings.

    Args:
        data_frame (DataFrame): Input dataframe containing predictions and user IDs.
        ground_truth (DataFrame): Ground truth ratings dataframe.
        k (int): Position to calculate NDCG at.

    Returns:
        float: Average NDCG.
    """
    ndcg_values = []

    for user_id in ground_truth.index:
        user_df = data_frame[data_frame["userid"] == user_id]\
            .sort_values(by="prediction", ascending=False)
        user_ground_truth = dict(zip(ground_truth.loc[user_id]["movieid"],
                                     ground_truth.loc[user_id]["rating"]))

        # Create a list of tuples (predicted rating, actual rating)
        ratings = [
            (getattr(row, "prediction"), user_ground_truth.get(getattr(row, "movieid"), None))
            for row in user_df.itertuples()
            if user_ground_truth.get(getattr(row, "movieid"), None) is not None
        ]

        # Sort the list by predicted rating
        ratings.sort(key=lambda x: x[0], reverse=True)

        # Calculate DCG
        dcg = sum(
            (2 ** actual_rating - 1) / np.log2(i + 2)
            for i, (_, actual_rating) in enumerate(ratings[:k])
        )

        # Calculate IDCG
        sorted_ratings = sorted([r[1] for r in ratings], reverse=True)
        idcg = sum(
            (2 ** actual_rating - 1) / np.log2(i + 2)
            for i, actual_rating in enumerate(sorted_ratings[:k])
        )

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        ndcg_values.append(ndcg)

    return np.mean(ndcg_values)
