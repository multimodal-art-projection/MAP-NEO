# -*- coding = utf-8 -*-
# @Time : 2024/1/8 22:46
# @Author : Cheng
# @File : Dolma_Filter.py
# @Software: PyCharm


import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import col
from util_with_try import filter_one_line_wrap
from pyspark.storagelevel import StorageLevel

CUR_CONTENT_FIELD_NAME = 'text'
NEW_CONTENT_FIELD_NAME = 'raw_content'

spark = SparkSession.builder \
            .appName("dolma filter") \
            .getOrCreate()
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO
    parser.add_argument("--input_dir", default="./dataset/dolma_cc_sampling.json",
                        help="Dataset input directory.")
    parser.add_argument("--output_dir", default="./cleaned_dataset",
                        help="Dataset output directory.")
    parser.add_argument("--content_field_name", default="text", help="Field name of the content.")
    # parser.add_argument("--output_file_final", help="File to output final docs to.")

    # gopher rules
    # TODO
    parser.add_argument("--gopher_stopwords_dir", default="./en_gopher_stopwords.json",
                        help="Gopher stop words file path.")
    parser.add_argument("--frac_stop_words_gopher", default=0.06, help="Fraction threshold of stop words in the document")
    parser.add_argument("--count_stop_words_gopher", default=2, help="Number threshold of stop words in the document")
    parser.add_argument("--no_alph_words_th", default=0.4,
                        help="The fraction of words that contain no alphabetical character.")
    parser.add_argument("--frac_symbol_words", default=0.5,
                        help="Fraction threshold of symbols to words in the content.Symbols are defined #, ..., and …")
    parser.add_argument("--threshold_rps_doc_mean_word_length_1", default=3)
    parser.add_argument("--threshold_rps_doc_mean_word_length_2", default=10)
    parser.add_argument("--threshold_rps_doc_frac_lines_end_with_ellipsis", default=0.2)
    parser.add_argument("--threshold_rps_lines_start_with_bulletpoint", default=0.9)

    # ccnet rules
    parser.add_argument("--ccnt_min_length_th", default=200, help="Minimum Number of characters")
    # TODO
    parser.add_argument("--fasttext_model_dir", default="./lid.176.bin",
                        help="Fasttext model directory")
    parser.add_argument("--threshold_fasttext_score", default=0.4, help="Lowest fasttext model score to keep")
    parser.add_argument("--theshold_num_lines", default=1)

    # c4 rules
    parser.add_argument("--min_word_count", default=50)
    parser.add_argument("--max_word_count", default=10000)
    parser.add_argument("--threshold_curly_braces", default=0.025)
    parser.add_argument("--threshold_lorem_ipsum", default=3e-8)
    parser.add_argument("--threshold_sentence_count", default=7500)

    # pretrainer's guide rules
    parser.add_argument("--threshold_frac_all_caps_words", default=0.1)
    parser.add_argument("--lower_threshold_frac_unique_words", default=0.1)
    parser.add_argument("--upper_threshold_frac_unique_words", default=0.8)

    # self-defined rules
    parser.add_argument("--threshold_num_sents", default=1)
    parser.add_argument("--frac_lines_endwith_readmore", default=0.1)
    parser.add_argument("--nonconsecutive_hash_ratio", default=0.1,
                        help="")
    parser.add_argument("--nonconsecutive_ellipsis_ratio", default=0.1,
                        help="")
    parser.add_argument("--threshold_frac_punct", default=0)
    parser.add_argument("--frac_words_no_alph_over_no_punct", default=0.2,
                        help="")
    parser.add_argument("--frac_words_digital_over_no_punct", default=0.3,
                        help="")

    # refined web rules
    parser.add_argument("--lower_threshold_doc_unigram_entropy", default=3)
    parser.add_argument("--upper_threshold_doc_unigram_entropy", default=6)
    parser.add_argument("--threshold_frac_chars_top2gram", default=0.20,
                        help="Threshold of fraction of characters contained within the most frequently-occurring 2-gram")
    parser.add_argument("--threshold_frac_chars_top3gram", default=0.18,
                        help="Threshold of fraction of characters contained within the most frequently-occurring 3-gram")
    parser.add_argument("--threshold_frac_chars_top4gram", default=0.16,
                        help="Threshold of fraction of characters contained within the most frequently-occurring 4-gram")
    parser.add_argument("--threshold_frac_chars_dupe5gram", default=0.15,
                        help="Threshold of fraction of characters contained within all duplicate 5-grams")
    parser.add_argument("--threshold_frac_chars_dupe6gram", default=0.14,
                        help="Threshold of fraction of characters contained within all duplicate 6-grams")
    parser.add_argument("--threshold_frac_chars_dupe7gram", default=0.13,
                        help="Threshold of fraction of characters contained within all duplicate 7-grams")
    parser.add_argument("--threshold_frac_chars_dupe8gram", default=0.12,
                        help="Threshold of fraction of characters contained within all duplicate 8-grams")
    parser.add_argument("--threshold_frac_chars_dupe9gram", default=0.11,
                        help="Threshold of fraction of characters contained within all duplicate 9-grams")
    parser.add_argument("--threshold_frac_chars_dupe10gram", default=0.10,
                        help="Threshold of fraction of characters contained within all duplicate 10-grams")
    parser.add_argument("--threshold_sents", default=0.90,
                        help="Similarity threshold between texts justified as duplicates")
    parser.add_argument("--threshold_frac_lines_dupe_lines", default=0.30,
                        help="Minimum proportion of duplicated sentences in the text to keep")
    parser.add_argument("--threshold_frac_chars_dupe_lines", default=0.20,
                        help="Minimum proportion of characters contained within those duplicated sentences in the text to keep")
    # TODO
    parser.add_argument("--ut1_url_blacklist_dir", default="./dest",
                        help="UT1 url blacklist filepath")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    CUR_CONTENT_FIELD_NAME = args.content_field_name
    df_cached = spark.read.json(input_dir).persist(StorageLevel.MEMORY_AND_DISK)
    print(f'before filter, the numbe of doc is {df_cached.count()}')
    udf_filter = udf(filter_one_line_wrap(args), BooleanType())
    df = df_cached.filter(udf_filter(col(CUR_CONTENT_FIELD_NAME)))
    df_cached.unpersist()
    df = df.withColumnRenamed(CUR_CONTENT_FIELD_NAME, NEW_CONTENT_FIELD_NAME).persist(StorageLevel.MEMORY_AND_DISK)
    df.write.option("compression", "gzip").json(output_dir)
    print(f'after filter, the numbe of doc is {df.count()}')
