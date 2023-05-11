# Data Characteristics

The datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation service. It
contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users
between January 09, 1995 and March 31, 2015. This dataset was generated on October 17, 2016. Users were selected at
random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user
is represented by an id, and no other information is provided.

The data consists of 5 groups:

rating.csv: contains ratings of movies by users:

- userId
- movieId
- rating
- timestamp

movie.csv: contains movie information:

- movieId
- title
- genres

tag.csv: contains tags applied to movies by users:

- userId
- movieId
- tag
- timestamp

genome_scores.csv: contains movie-tag relevance data:

- movieId
- tagId
- relevance

genome_tags.csv: contains tag descriptions:

- tagId
- tag

# How to download

```shell
wget https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/3EHBfJGmiv--MA -O data/genome_scores.csv
wget https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/Go-JBXBqKtk6Vg -O data/genome_tags.csv
wget https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/QPo2T-LjytuHMg -O data/movie.csv
wget https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/sFD-ql1stXijKg -O data/rating.csv
wget https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/i/1ZVz6Jq5RzThaQ -O data/tag.csv
```
