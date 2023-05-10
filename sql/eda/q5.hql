USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q5'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT movieId, title, avg_rating, rating_count
FROM (
  SELECT m.movieId, m.title, AVG(r.rating) AS avg_rating, COUNT(r.rating) AS rating_count
  FROM movie AS m
  JOIN rating AS r ON m.movieId = r.movieId
  GROUP BY m.movieId, m.title
  HAVING COUNT(r.rating) >= 100
) subquery
ORDER BY avg_rating DESC
LIMIT 10;

