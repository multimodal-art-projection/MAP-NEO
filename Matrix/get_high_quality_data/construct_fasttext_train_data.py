import argparse
import math
import os
import random
from pyspark.sql import SparkSession
from urllib.parse import urlparse

parser = argparse.ArgumentParser()
parser.add_argument('--neg_file_path', type=str)
parser.add_argument('--pos_file_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()
neg_file_path = args.neg_file_path
pos_file_path = args.pos_file_path
output_path = args.output_path
num_sample = 500000
num_partition = 1000


spark = SparkSession.builder \
            .appName("construct spark train data") \
            .getOrCreate()
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")


def get_base_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

def sample_from_group(url_texts, total_samples, total_groups_size):
    sample_num = int(math.ceil(total_samples / total_groups_size))
    if len(url_texts) < sample_num:
        return url_texts
    return random.sample(url_texts, sample_num)


neg_df = spark.read.json(neg_file_path).repartition(num_partition).cache()
neg_df_num = neg_df.count()
sample_fraction = min(1.0, num_sample / neg_df_num)
sampled_neg_df = neg_df.sample(False, sample_fraction).cache()
sampled_neg_df.write.json(os.path.join(output_path,'neg'))
print(f'neg num: {sampled_neg_df.count()}')

pos_rdd = spark.read.json(pos_file_path).repartition(num_partition).rdd.map(lambda x: (x['url'], x['raw_content'])).cache()
pos_rdd_num = pos_rdd.count()
base_url_rdd = pos_rdd.map(lambda x: (get_base_url(x[0]), (x[0], x[1])))
grouped_by_base_url_rdd = base_url_rdd.groupByKey().cache()

group_size = grouped_by_base_url_rdd.count()
sampled_pos_rdd = grouped_by_base_url_rdd.flatMapValues(lambda x: sample_from_group(list(x), num_sample, group_size)).values()
sampled_pos_df = sampled_pos_rdd.toDF(['url', 'raw_content']).cache()
sampled_pos_df.write.json(os.path.join(output_path,'pos'))
print(f'pos num: {sampled_pos_df.count()}')
