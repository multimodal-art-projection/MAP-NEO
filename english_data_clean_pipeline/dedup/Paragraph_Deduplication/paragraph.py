import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
CONTENT_FIELD_NAME = 'raw_content'


def paragraph_dedup(spark, df, from_column, to_column, min_len=0, id_column=""):
    drop_id_column = False
    df = df.withColumn("paragraphs", F.split(F.col(from_column), "\n"))
    df = df.select(id_column, F.posexplode(F.col("paragraphs")).alias("pos", "paragraph"))
    df = df.withColumn("hash", F.sha2("paragraph", 256))  # 256-bit hash
    # Groups by hash, sorts by (id, pos), then selects the 1st row in each group.
    window = Window.partitionBy("hash").orderBy(id_column, "pos")
    df = df.withColumn("rn", F.row_number().over(window))
    df = df.where(F.col("rn") == 1).drop("rn")

    # Groups by id and collects "paragraph"s into a list, sorted by "pos".
    df = df.orderBy(id_column, "pos").groupBy(id_column).agg(F.collect_list("paragraph").alias("paragraphs"))
    df = df.withColumn(to_column, F.concat_ws("\n", F.col("paragraphs"))).drop("paragraphs")
    if drop_id_column: df = df.drop(id_column)
    if min_len > 0: df = df.filter(F.length(F.col(to_column)) >= min_len)
    return df

if __name__ == '__main__':
    spark = SparkSession.builder.appName("paragraph dedup").getOrCreate()
    df = spark.read.json(input_dir)
    df = paragraph_dedup(spark, df, from_column=CONTENT_FIELD_NAME, to_column=CONTENT_FIELD_NAME, id_column='doc_id', min_len=20)
    print('after paragraph dedup: %d' % df.count())
    df.write.option("compression", "gzip").json(output_dir)