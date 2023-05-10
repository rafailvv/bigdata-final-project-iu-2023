USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY '/root/q4'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT m.genres, AVG(r.rating) as avg_rating
FROM movie m JOIN rating r
ON m.movieId = r.movieId
GROUP BY m.genres
ORDER BY avg_rating DESC;



