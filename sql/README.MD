# Data Collection and Ingestion

- `sql/tables_creation.sql` is a `.sql` file which built all tables from dataset in the database, add all constraints
  and load the data to the tables from `.csv` files
- `sql/hive_tables_creation.hql` is a `.hql` create Hive Tables and stored the data
- 
# Exploratory Data Analysis

- `sql/eda` is a folder with `.hql` queries from q1 to q5

**Q1: Distribution of genres**

This query analyzes the distribution of genres across the entire Movie Lens dataset. It gives us an overview of the most
and least common movie genres, which can be useful for understanding the overall makeup of the dataset. This could be a
key insight if we consider that user preferences for genres could significantly influence their movie ratings.
Therefore, understanding the genre distribution can help us fine-tune our recommendation system and potentially
highlight genre biases in the dataset.

**Q2: Average rating per movie**

This query calculates the average rating for each movie in the dataset. This information is crucial for understanding
how well-received each movie is overall. A movie with a higher average rating is generally more favored by the audience,
and such movies can be prioritized in recommendations. However, it's also important to consider the number of ratings a
movie has received, as a high average rating based on a small number of reviews might not be as reliable as a slightly
lower average rating based on a large number of reviews.

**Q3: Number of ratings per movie**

This query determines the number of ratings each movie in the dataset has received. This is important because it gives
us an idea of a movie's popularity and the reliability of its average rating. A movie with a large number of ratings is
likely to have a more reliable average rating. Moreover, highly-rated movies that have also received a lot of ratings
can be considered popular favorites and could be given higher priority in a recommendation system.

**Q4: Average rating of movies per genre**

This query computes the average rating for each genre in the dataset. This information can help us understand which
genres are generally more well-received by the audience. A genre with a higher average rating might be more favored, and
movies from these genres could be recommended more often. It is important to remember that individual preferences can
vary widely, so genre preferences should be combined with other user-specific information to make accurate
recommendations.

**Q5: Top 10 movies with the highest average rating and at least 100 ratings**

This query lists the top 10 movies that not only have the highest average ratings but also have received at least 100
ratings. The 100 ratings threshold helps ensure the reliability of the average rating, as it filters out movies that
might have a high average rating based on a small number of ratings. The movies identified by this query could be
considered highly recommended by users, and the recommendation system might prioritize suggesting these movies to users
with similar tastes.