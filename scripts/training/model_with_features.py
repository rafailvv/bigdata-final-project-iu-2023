"""
This module trains a recommendation model using ALS algorithm with gener feature
cand evaluates it using regression metrics.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
import pandas as pd

# Initialize SparkSession
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

movies = spark.read.format("avro").table("projectdb.movie")

# Convert genres to numeric
indexer = StringIndexer(inputCol="genres", outputCol="genresIndex")

# One-hot encode the genres
encoder = OneHotEncoder(inputCol="genresIndex", outputCol="genresVec")

# Define the pipeline
pipeline = Pipeline(stages=[indexer, encoder])

# Fit and transform the data
movies = pipeline.fit(movies).transform(movies)

# Merge rating and movie data
data = rating.join(movies, on="movieid")

(training_als, test_als) = rating.randomSplit([0.7, 0.3], seed=42)

# Split the data into training and test sets
(training, test) = data.randomSplit([0.7, 0.3], seed=42)

crossval = get_crossval()

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training_als)
bestModel = cvModel.bestModel
# Use the model to make predictions on the test data
predictions_als = bestModel.transform(test)

# Define a Random Forest model
rf = RandomForestRegressor(featuresCol="genresVec", labelCol="rating")

# Build the pipeline
pipeline = Pipeline(stages=[rf])

# Train the model
model_features = pipeline.fit(training)

# Make predictions using the content-based model
predictions_features = model_features.transform(test)

# Convert the predictions to a DataFrame
predictions_als = predictions_als.toPandas()
predictions_features = predictions_features.toPandas()

# Rename the prediction columns
predictions_als = predictions_als.rename(columns={"prediction": "prediction_als"})
predictions_features = predictions_features.rename(
    columns={"prediction": "prediction_features"}
)

# Merge the two DataFrames on the movie ID and user ID
predictions = pd.merge(predictions_als, predictions_features, on=["movieid", "userid"])

# Calculate the final prediction as a weighted average of the two predictions
predictions["final_prediction"] = (
    0.5 * predictions["prediction_als"] + 0.5 * predictions["prediction_features"]
)
predictions = predictions.rename(
    columns={
        "rating_x": "rating",
        "final_prediction": "prediction",
        "title_x": "title",
        "timestamp_x": "timestamp",
        "genres_x": "genres",
    }
)
predictions = predictions.rename(columns={"final_prediction": "prediction"})
predictions = predictions[
    ["timestamp", "userid", "movieid", "title", "genres", "rating", "prediction"]
]
predictions = predictions.dropna()

predictions.to_csv("ast_features_predictions.csv")

# Group the DataFrame by user ID and collect the movie
# IDs and ratings for each user into a dictionary
ground_truth_dict = {}
for user_id, group in predictions.groupby("userid"):
    movie_ids = group["movieid"].tolist()
    ratings = group["rating"].tolist()
    ground_truth_dict[user_id] = {"movieid": movie_ids, "rating": ratings}

# Convert the dictionary to a DataFrame and set the user ID as the index
ground_truth = pd.DataFrame.from_dict(ground_truth_dict, orient="index")

print("mAP: " + str(calculate_map(predictions, ground_truth)))
print("NDCG: " + str(calculate_ndcg(predictions, ground_truth, k=3)))
