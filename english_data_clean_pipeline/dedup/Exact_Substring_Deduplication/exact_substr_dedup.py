import argparse
import hashlib
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()
duplicate_len = 50
min_length = 20
num_partition = 39000
input_dir = args.input_dir
output_dir = args.output_dir


CONTENT_FIELD_NAME = 'raw_content'
DOC_ID_FIELD_NAME = 'doc_id'
spark = SparkSession.builder \
        .appName("new exact dedup spark session") \
        .getOrCreate()

def _get_hash(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)

def text2tokens_wrap(tokenizer):
    def text2tokens(item):
        text, doc_id = item
        tokens = tokenizer.encode(text)
        return (doc_id, tokens)
    return text2tokens

# the inpute format is (new_doc_id, (text, ori_doc_id))
def text2hash_wrap(duplicate_len):
    def text2hash(item):
        doc_id, (text, _) = item
        ret = []
        for offset in range(len(text) - duplicate_len):
            hash_val = _get_hash(text[offset:offset+duplicate_len])
            ret.append((hash_val, (doc_id, offset)))
        return ret
    return text2hash

def get_remove_list(item):
    _, dup_list = item
    dup_list = list(dup_list)
    return dup_list[1:]

def read_json(input_dir):
    df = spark.read.json(input_dir)
    df = df.select(CONTENT_FIELD_NAME, DOC_ID_FIELD_NAME)
    return df.rdd.map(lambda row: (row[CONTENT_FIELD_NAME], row[DOC_ID_FIELD_NAME]))

# the input format is (doc_id, ((text, ori_doc_id), [offsets]))
def remove_dup_content_wrap(duplicate_len):
    def remove_dup_content(item):
        doc_id, ((text, ori_doc_id), offsets) = item
        remove_index = set()
        if offsets != None:
            for offset in offsets:
                for dup_offset in range(offset, offset+duplicate_len):
                    remove_index.add(dup_offset)
            cleaned_text = ''
            for idx, char in enumerate(text):
                if idx not in remove_index:
                    cleaned_text += char
            return (ori_doc_id, cleaned_text)
        else:
            return (ori_doc_id, text)
    return remove_dup_content

# load data, get (text, ori_doc_id)
rdd = read_json(input_dir)
# the added index is doc_id, get ((text, ori_doc_id), new_doc_id)
rdd = rdd.zipWithIndex()

# get (new_doc_id, (text, ori_doc_id))
rdd_with_index = rdd.map(lambda x: (x[1], x[0])).persist(StorageLevel.MEMORY_AND_DISK)
print("the number of docs is ", rdd_with_index.count())
# hash its value using a window size of duplicate_len, get (hash_val, (new_doc_id, offset))
rdd = rdd_with_index.flatMap(text2hash_wrap(duplicate_len))
# after groupByKey(), we get (hash_val, [(new_doc_id, offset)]) we use data[1][1:] to get the remove list, whose format is (new_doc_id, offset)
rdd = rdd.groupByKey().flatMap(get_remove_list)
# get (new_doc_id, [offset])
rdd = rdd.groupByKey()
# rdd_tokens leftOuterJoin With rdd, get (doc_id, ((text, ori_doc_id ), [offsets]))
rdd = rdd_with_index.leftOuterJoin(rdd)
# remove duplicate tokens, get (or_doc_id, text)
rdd = rdd.map(remove_dup_content_wrap(duplicate_len))
rdd = rdd.filter(lambda x: len(x[1]) >= min_length)
# transfrom rdd to dataframe
df = rdd.toDF([DOC_ID_FIELD_NAME, CONTENT_FIELD_NAME])
# save as jsonl format
df.write.option("compression", "gzip").json(output_dir)
print("the number of deduped docs is ", df.count())