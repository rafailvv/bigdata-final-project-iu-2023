#!/bin/bash

hdfs dfs -mkdir -p /project/avsc
hdfs dfs -put *.avsc /project/avsc

hive -f sql/hive_tables_creation.hql >hive_results.txt

hive -f sql/eda/q1.hql
echo "genres,count" >output/eda/q1.csv
cat /root/q1/* >>output/eda/q1.csv

hive -f sql/eda/q2.hql
echo "movieId,title,avg_rating" >output/eda/q2.csv
cat /root/q2/* >>output/eda/q2.csv

hive -f sql/eda/q3.hql
echo "movieId,title,rating_count" >output/eda/q3.csv
cat /root/q3/* >>output/eda/q3.csv

hive -f sql/eda/q4.hql
echo "genres,avg_rating" >output/eda/q4.csv
cat /root/q4/* >>output/eda/q4.csv

hive -f sql/eda/q5.hql
echo "movieId|title|avg_rating|rating_count" >output/eda/q5.csv
cat /root/q5/* >>output/eda/q5.csv
