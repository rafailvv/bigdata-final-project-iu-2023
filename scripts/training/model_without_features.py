from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import pandas as pd


def calculate_mAP(df, ground_truth):
    user_aps = []

    for user_id in ground_truth.index:
        user_df = df[df['userid'] == user_id].sort_values(by='prediction', ascending=False)
        num_correct = 0
        precision = []
        recall = []

        for i in range(len(user_df)):
            if user_df.iloc[i]['movieid'] in ground_truth.loc[user_id]['movieid']:
                num_correct += 1
            precision.append(num_correct / (i + 1))
            recall.append(num_correct / len(ground_truth.loc[user_id]['movieid']))

        # Calculate average precision
        ap = 0
        for i in range(len(precision)):
            if i == 0 or recall[i] != recall[i - 1]:
                ap += precision[i] * (recall[i] - recall[i - 1] if i > 0 else recall[i])

        user_ap = ap / len(ground_truth.loc[user_id]['movieid'])
        user_aps.append(user_ap)

    return np.mean(user_aps)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def calculate_ndcg(df, ground_truth, k):
    ndcg_values = []

    for user_id in ground_truth.index:
        user_df = df[df['userid'] == user_id].sort_values(by='prediction', ascending=False)
        user_ground_truth = dict(zip(ground_truth.loc[user_id]['movieid'], ground_truth.loc[user_id]['rating']))

        # Create a list of tuples (predicted rating, actual rating)
        ratings = []
        for row in user_df.itertuples():
            predicted_rating = getattr(row, 'prediction')
            actual_rating = user_ground_truth.get(getattr(row, 'movieid'), None)
            if actual_rating is not None:
                ratings.append((predicted_rating, actual_rating))

        # Sort the list by predicted rating
        ratings.sort(key=lambda x: x[0], reverse=True)

        # Calculate DCG and IDCG
        dcg, idcg = 0.0, 0.0
        for i, (predicted_rating, actual_rating) in enumerate(ratings):
            if i < k:
                dcg += (2 ** actual_rating - 1) / np.log2(i + 2)  # We use i+2 because i is 0-indexed

        sorted_ratings = sorted([r[1] for r in ratings], reverse=True)
        for i, actual_rating in enumerate(sorted_ratings):
            if i < k:
                idcg += (2 ** actual_rating - 1) / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        ndcg_values.append(ndcg)

    return np.mean(ndcg_values)


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("BDT Project") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083") \
    .config("spark.sql.avro.compression.codec", "snappy") \
    .enableHiveSupport() \
    .getOrCreate()

# Load the data
rating = spark.read.format("avro").table('projectdb.rating')
rating = rating.sample(100000.0 / rating.count(), seed=42)

# Split the data into training and test sets
(training, test) = rating.randomSplit([0.7, 0.3], seed=42)

# Define a grid for the hyperparameter search
als = ALS(maxIter=5, userCol="userid", itemCol="movieid", ratingCol="rating", coldStartStrategy="drop")
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 50, 100]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .build()

# Define a regression evaluator that will be used to evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

# Cross validation over the parameter grid
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          parallelism=3,
                          seed=42,
                          numFolds=4)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)
bestModel = cvModel.bestModel
# Use the model to make predictions on the test data
predictions = bestModel.transform(test)

# Convert the PySpark DataFrame to a Pandas DataFrame
df = predictions.toPandas()
df.to_csv("ast_predictions.csv")

# Group the DataFrame by user ID and collect the movie IDs and ratings for each user into a dictionary
ground_truth_dict = {}
for user_id, group in df.groupby("userid"):
    movie_ids = group["movieid"].tolist()
    ratings = group["rating"].tolist()
    ground_truth_dict[user_id] = {"movieid": movie_ids, "rating": ratings}

# Convert the dictionary to a DataFrame and set the user ID as the index
ground_truth = pd.DataFrame.from_dict(ground_truth_dict, orient="index")

print("mAP: " + str(calculate_mAP(df, ground_truth)))
print("NDCG: " + str(calculate_ndcg(df, ground_truth, k=3)))
