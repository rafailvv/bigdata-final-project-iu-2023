DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;

SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec

CREATE EXTERNAL TABLE tag STORED AS AVRO LOCATION '/project/tag' TBLPROPERTIES ('avro.schema.url'='/project/avsc/tag.avsc');
CREATE EXTERNAL TABLE rating STORED AS AVRO LOCATION '/project/rating' TBLPROPERTIES ('avro.schema.url'='/project/avsc/rating.avsc');
CREATE EXTERNAL TABLE movie STORED AS AVRO LOCATION '/project/movie' TBLPROPERTIES ('avro.schema.url'='/project/avsc/movie.avsc');
CREATE EXTERNAL TABLE genome_scores STORED AS AVRO LOCATION '/project/genome_scores' TBLPROPERTIES ('avro.schema.url'='/project/avsc/genome_scores.avsc');
CREATE EXTERNAL TABLE genome_tags STORED AS AVRO LOCATION '/project/genome_tags' TBLPROPERTIES ('avro.schema.url'='/project/avsc/genome_tags.avsc');

SELECT COUNT(*) FROM movie


