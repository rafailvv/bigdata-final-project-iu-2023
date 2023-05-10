-- switch to the database
\c project;

-- Optional
START TRANSACTION;

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

BEGIN;
CREATE TABLE tag
(
    userId     INT,
    movieId    INT,
    tag        VARCHAR(255),
    _timestamp timestamp,
    temp       TEXT
);

CREATE TABLE rating
(
    userId    INT,
    movieId   INT,
    rating    FLOAT,
    _timestamp timestamp,
    temp TEXT
);

CREATE TABLE movie
(
    movieId INT PRIMARY KEY,
    title   VARCHAR(255),
    genres  VARCHAR(255)
);

CREATE TABLE genome_scores
(
    movieId   INT,
    tagId     INT,
    relevance FLOAT
);

CREATE TABLE genome_tags
(
    tagId INT PRIMARY KEY,
    tag   VARCHAR(255)
);

ALTER TABLE tag ADD CONSTRAINT pk_tag PRIMARY KEY (userId, movieId, tag);
ALTER TABLE rating ADD CONSTRAINT pk_rating PRIMARY KEY (userId, movieId);
ALTER TABLE movie ALTER COLUMN movieId SET NOT NULL;
ALTER TABLE genome_scores ADD CONSTRAINT fk_genome_scores_movie FOREIGN KEY (movieId) REFERENCES movie(movieId);
ALTER TABLE genome_scores ADD CONSTRAINT fk_genome_scores_tag FOREIGN KEY (tagId) REFERENCES genome_tags(tagId);


\COPY tag(userId, movieId, tag, temp) FROM 'data/tag.csv' DELIMITER ',' CSV HEADER NULL AS 'null';
UPDATE tag SET _timestamp = to_date(temp, 'YYYY-MM-DD HH24:MI:SS')::timestamp;
ALTER TABLE tag DROP COLUMN temp;

\COPY rating(userId, movieId, rating, temp) FROM 'data/rating.csv' DELIMITER ',' CSV HEADER NULL AS 'null';
UPDATE rating SET _timestamp = to_date(temp, 'YYYY-MM-DD HH24:MI:SS')::timestamp;
ALTER TABLE rating DROP COLUMN temp;

\COPY movie FROM 'data/movie.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

\COPY genome_tags FROM 'data/genome_tags.csv' DELIMITER ',' CSV HEADER NULL AS 'null';
\COPY genome_scores FROM 'data/genome_scores.csv' DELIMITER ',' CSV HEADER NULL AS 'null';

COMMIT;

SELECT COUNT(*) FROM movie;