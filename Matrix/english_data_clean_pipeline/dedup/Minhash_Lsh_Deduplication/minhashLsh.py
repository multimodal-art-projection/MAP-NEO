import argparse
import ftfy 
import re
import string

from datasketch import MinHash
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession
from nltk import ngrams

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='input directories')
parser.add_argument('--inter_dir', help='output directory')
parser.add_argument('--output_dir', help='output directory')
parser.add_argument('--content_field_name', help='content field name', default='raw_content')
args = parser.parse_args()
width = 13
bands = 9
r = 13
n_perm = 128
num_partition = 8000
input_dir = args.input_dir
output_dir = args.output_dir
inter_dir = args.inter_dir
CONTENT_FIELD_NAME = args.content_field_name

# transform array to byte representation so that it can be hashed
def _H(hs):
    return bytes(hs.byteswap().data)

def normalize_text(pair):
    text = pair[0]
    text = ftfy.fix_text(text, normalization="NFC")
    # lower cased
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return (text, pair[1])

# (todo(panding): find an efficent way to compute minhash)
def generate_hash_values(pair):
    global width
    global r
    global bands
    text = pair[0]
    features = map(lambda x: "".join(x), ngrams(text, width))
    m = MinHash(num_perm=128)
    [m.update(x.encode('utf8')) for x in features]
    ret = []
    for idx in range(bands):
        ret.append((idx, _H(m.hashvalues[idx * r: (idx+1) * r]), pair[1]))
    return ret

def generate_minhash_value(pair):
    global width
    global r
    global bands
    text = pair[0]
    features = map(lambda x: "".join(x), ngrams(text, width))
    m = MinHash(num_perm=128)
    [m.update(x.encode('utf8')) for x in features]
    return (m.hashvalues, pair[1])

def generate_bands(pair):
    hashvalues = pair[0]
    ret = []
    for idx in range(bands):
        ret.append((idx, _H(hashvalues[idx * r: (idx+1) * r]), pair[1]))
    return ret

def generate_edges(doc_id_list):
    edges = []
    for i in range(len(doc_id_list) - 1):
        if (doc_id_list[i] > doc_id_list[i+1]):
            edges.append((doc_id_list[i], doc_id_list[i+1]))
        else:
            edges.append((doc_id_list[i+1], doc_id_list[i]))
    return edges

def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]

def large_star_reduce(pair):
    u = pair[0]
    points = pair[1]
    min_point = u
    for point in points:
        min_point = min(min_point, point)
    ret = []
    for point in points:
        if point > u:
            ret.append((point, min_point))
    return ret

def small_star_map(pair):
    u, v = pair
    if v <= u:
        return (u, v)
    else:
        return (v, u)

def small_star_reduce(pair):
    u, N = pair
    m = u
    for point in N:
        m = min(m, point)
    ret = []
    if m != u:
        ret.append((u, m))
    for point in N:
        if point != m:
            ret.append((point, m))
    return ret

spark = SparkSession.builder.appName("minhashLsh").getOrCreate()

def minhashLsh(input_dir, output_dir):
    df = spark.read.json(input_dir)
    df1 = df.repartition(num_partition)
    # normalize text
    rdd = df1.rdd.map(lambda row: (row[CONTENT_FIELD_NAME], row["doc_id"]))
    rdd = rdd.filter(lambda x: x[0] != None)
    rdd = rdd.map(normalize_text)
    # print("finish normalize text")
    # generate hash value
    rdd_cache = rdd.map(generate_minhash_value).persist(StorageLevel.MEMORY_AND_DISK)
    rdd_cache.saveAsPickleFile(inter_dir)
    # rdd = rdd.flatMap(generate_hash_values)
    # rdd.saveAsTextFile(inter_dir)
    print("finish generate hash value")
    rdd = rdd_cache.flatMap(generate_bands)
    rdd_cache.unpersist()
    # find duplicate pairs which is represented by edges
    edges = rdd.groupBy(lambda x: (x[0], x[1])).flatMap(lambda x: generate_edges([i[2] for i in x[1]])).distinct().cache()
    # find connect components, the reference can be seen in https://dl.acm.org/doi/10.1145/2670979.2670997
    a = edges
    while True:
        b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
        a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
        changes = a.subtract(b).union(b.subtract(a)).count()
        if changes == 0:
            break
    remove_list = a.map(lambda x: (x[0],)).toDF(['doc_id'])
    # print("the number of distinct remove doc id: ", remove_list.distinct().count())
    n_remove = remove_list.count()
    print("the number of doc need to be remove is ", n_remove)
    df = df.join(remove_list, on='doc_id', how="left_anti")
    df.write.option("compression", "gzip").json(output_dir)

    

if __name__ == '__main__':
    minhashLsh(input_dir, output_dir)