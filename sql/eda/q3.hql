USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q3'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT movieId, title, rating_count
FROM (
  SELECT m.movieId, m.title, COUNT(r.rating) AS rating_count
  FROM movie AS m
  JOIN rating AS r ON m.movieId = r.movieId
  GROUP BY m.movieId, m.title
) subquery
ORDER BY rating_count DESC;

