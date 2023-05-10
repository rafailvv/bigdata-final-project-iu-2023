#!/bin/bash

psql -U postgres -d project -f sql/tables_creation.sql

wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar --no-check-certificate

cp postgresql-42.6.0.jar /usr/hdp/current/sqoop-client/lib/

sqoop list-databases --connect jdbc:postgresql://localhost/postgres --username postgres

sqoop list-tables --connect jdbc:postgresql://localhost/project --username postgres

sqoop eval --connect jdbc:postgresql://localhost/project --username postgres --query "SELECT * FROM movie LIMIT 10"

sqoop import-all-tables -Dmapreduce.job.user.classpath.first=true --connect jdbc:postgresql://localhost/project --username postgres --warehouse-dir /project --as-avrodatafile --compression-codec=snappy --outdir /project/avsc --m 1
