"""
Additional features for the model
"""

from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

def get_crossval():
    als = ALS(
        maxIter=5,
        userCol="userid",
        itemCol="movieid",
        ratingCol="rating",
        coldStartStrategy="drop",
    )
    paramGrid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 50, 100])
        .addGrid(als.regParam, [0.01, 0.1, 1.0])
        .build()
    )

    # Define a regression evaluator that will be used to evaluate the model
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    # Cross validation over the parameter grid
    crossval = CrossValidator(
        estimator=als,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        parallelism=3,
        seed=42,
        numFolds=4,
    )
    return crossval
