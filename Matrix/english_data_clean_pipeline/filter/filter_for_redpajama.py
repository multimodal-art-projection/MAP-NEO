import argparse
import io
import json
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
from pyspark.sql import Row
from util import filter_func

python_version = sys.version
print(python_version)

doc_file_name = sys.argv[1].strip()
output_dir_name = doc_file_name[:-9]
input_dir_name = doc_file_name.split('_')[0]
signal_file_name = sys.argv[2].strip()

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser
# input_dir = '/home/panding/DataPipeline/small_dataset'
output_dir = f'us3://redpajamav2/dataClean/life/{output_dir_name}'
prefix = f"us3://redpajamav2/download/rawdata/{input_dir_name}/redpajamav2"
prefix_signal = 'us3://redpajamav2/quality_signals/redpajamav2'

print('output_dir: ', output_dir)
print('prefix: ', prefix)
print('prefix_signal', prefix_signal)


file_paths = []
with open(doc_file_name, 'r') as f:
    for idx, file_name in enumerate(f.readlines()):
        file_name = file_name.strip()
        file_paths.append(os.path.join(prefix, file_name))
signal_paths = []
with open(signal_file_name, 'r') as f:
    for idx, signal_name in enumerate(f.readlines()):
        signal_name = signal_name.strip()
        signal_paths.append(os.path.join(prefix_signal, signal_name))

app_name = f'signal filter {output_dir_name}'  
spark = SparkSession.builder \
    .appName(app_name) \
    .getOrCreate()
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")

def extractJsonLine(item):
    wholeText = item.value
    file_path = item.file_path
    file = io.StringIO(wholeText)
    ret = []
    for line_num, line in enumerate(file):
        try:
            json_obj = json.loads(line)
            file_name = file_path.split('/')[-1]
            components = file_name.split('__')[-3:]
            shard_id = '/'.join(components)
            doc_id = os.path.join(shard_id, str(line_num))
            json_obj['doc_id'] = doc_id
            ret.append(json_obj)
        except json.JSONDecodeError:
            pass
    return ret

# read jsonl file
doc_df = spark.read.text(file_paths, wholetext=True).withColumn("file_path", input_file_name())
doc_rdd = doc_df.rdd
# here, we get doc_rdd with doc_id
doc_rdd = doc_rdd.flatMap(extractJsonLine)
doc_rdd = doc_rdd.map(lambda x: (x['doc_id'], x['raw_content']))

# read signal file
signal_df = spark.read.json(signal_paths)
signal_rdd = signal_df.rdd.map(lambda x: (x['id'], x['quality_signals']))

# join doc_rdd and signal_rdd, rdd = (doc_id, (doc_dic, signal_dic))
rdd = doc_rdd.join(signal_rdd)
rdd = rdd.map(lambda x: {'doc_id': x[0], 'raw_content': x[1][0], **x[1][1].asDict()})

# print('before filter, the number of document is ', rdd.count())

rdd = rdd.filter(filter_func)
df = spark.createDataFrame(rdd)
df.write.option("compression", "gzip").json(output_dir)