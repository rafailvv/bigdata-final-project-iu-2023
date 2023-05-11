"""
This module trains a recommendation model using ALS algorithm
cand evaluates it using regression metrics.
"""

from pyspark.sql import SparkSession
import pandas as pd

from scripts.training.map_calculator import calculate_map
from scripts.training.misc import get_crossval
from scripts.training.ndcg_calculator import calculate_ndcg

spark = (
    SparkSession.builder.appName("BDT Project")
    .config("spark.sql.catalogImplementation", "hive")
    .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")
    .config("spark.sql.avro.compression.codec", "snappy")
    .enableHiveSupport()
    .getOrCreate()
)

# Load the data
rating = spark.read.format("avro").table("projectdb.rating")
rating = rating.sample(100000.0 / rating.count(), seed=42)

# Split the data into training and test sets
(training, test) = rating.randomSplit([0.7, 0.3], seed=42)

crossval = get_crossval()

# Run cross-validation and choose the best set of parameters
cvModel = crossval.fit(training)
bestModel = cvModel.bestModel

# Use the model to make predictions on the test data
predictions = bestModel.transform(test)

# Convert the PySpark DataFrame to a Pandas DataFrame
df = predictions.toPandas()
df.to_csv("ast_predictions.csv")

# Group the DataFrame by user ID and collect the movie IDs and ratings for each user
ground_truth_dict = {}
for user_id, group in df.groupby("userid"):
    movie_ids = group["movieid"].tolist()
    ratings = group["rating"].tolist()
    ground_truth_dict[user_id] = {"movieid": movie_ids, "rating": ratings}

# Convert the dictionary to a DataFrame and set the user ID as the index
ground_truth = pd.DataFrame.from_dict(ground_truth_dict, orient="index")

print("mAP: " + str(calculate_map(df, ground_truth)))
print("NDCG: " + str(calculate_ndcg(df, ground_truth, k=3)))
