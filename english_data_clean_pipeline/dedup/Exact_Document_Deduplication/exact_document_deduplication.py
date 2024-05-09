import argparse
from pyspark.sql import SparkSession

CONTENT_FIELD_NAME = "raw_content"
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='input directories')
parser.add_argument('--output_dir', help='output directory')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

spark = SparkSession.builder \
        .appName("new exact dedup spark session") \
        .getOrCreate()

df = spark.read.json(input_dir)
df = df.dropDuplicates([CONTENT_FIELD_NAME])
df.write.option("compression", "gzip").json(output_dir)