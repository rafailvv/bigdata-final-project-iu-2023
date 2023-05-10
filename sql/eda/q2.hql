USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT movieId, title, avg_rating
FROM (
  SELECT m.movieId, m.title, AVG(r.rating) AS avg_rating
  FROM movie AS m
  JOIN rating AS r ON m.movieId = r.movieId
  GROUP BY m.movieId, m.title
) subquery
ORDER BY avg_rating DESC;
