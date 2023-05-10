USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q1'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT genres, genre_count
FROM (
  SELECT genres, COUNT(*) AS genre_count
  FROM movie
  GROUP BY genres
) subquery
ORDER BY genre_count DESC;
