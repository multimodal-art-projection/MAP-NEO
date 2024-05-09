import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
from pyspark import StorageLevel
from utils import compute_confidence

# Initialize SparkSession
spark = SparkSession.builder \
            .appName("Filter Math Data Test Threshold") \
            .getOrCreate()
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir_path', type=str)
parser.add_argument('--output_dir_path', type=str)
parser.add_argument('--model_path', type=str, default='model.bin')
parser.add_argument('--threshhold', type=float, default=0.96)
args = parser.parse_args()
model_path = args.model_path
threshhold = args.threshhold
input_dir_path = args.input_dir_path
output_dir_path =  args.output_dir_path

# Register UDF
compute_confidence_udf = udf(compute_confidence, FloatType())

# Load data into DataFrame
df = spark.read.json(input_dir_path)

# Apply UDF to DataFrame
filtered_df = df.withColumn("confidence", compute_confidence_udf(col("raw_content"))).persist(StorageLevel.MEMORY_AND_DISK)

# At this point, you can filter the DataFrame based on the confidence score
df_filtered = filtered_df.filter(col("confidence") > threshhold)
df_filtered.write.json(output_dir_path)