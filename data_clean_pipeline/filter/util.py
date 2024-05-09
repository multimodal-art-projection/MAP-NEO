# -*- coding = utf-8 -*-
# @Time : 2023/12/29 15:32
# @Author : Cheng
# @File : Docu_Signal_Filter.py
# @Software: PyCharm
import argparse
import nltk
import numpy as np
import string
import os
import re
import json
import zipfile
from bad_url_words import STRICT_BAD_URL_WORDS, HARD_BAD_URL_WORDS
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cwd = os.getcwd()
# 解压缩 nltk_data.zip
print('current path is ', cwd)
for file_name in os.listdir(cwd):
    print(file_name)
print("in nltk data, the file is ")
subdir = os.path.join(cwd, 'nltk_data')
for file_name in os.listdir(subdir):
    print(file_name)
print("in nltk_data of nltk_data, the file is ")
subsubdir = os.path.join(subdir, 'nltk_data')
for file_name in os.listdir(subsubdir):
    print(file_name)
# with zipfile.ZipFile(os.path.join(cwd, "nltk_data.zip"), 'r') as zip_ref:
#     zip_ref.extractall(cwd)
nltk.data.path.append(subsubdir)

gopher_stopwords_dir = 'en_gopher_stopwords.json'

def gopher_rules_pass(line, signals) -> bool:
    """ function returns True if the sample complies with Gopher rules """

    # rule 1: mean word length between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # rule 2: fraction of lines that end with an ellipsis must be below 0.2
    lines_end_with_ellipsis = signals["rps_doc_frac_lines_end_with_ellipsis"][0][2]
    if lines_end_with_ellipsis >= 0.2:
        return False

    # rule 3: 90% of lines need to start without a bullet point
    n_lines = signals["ccnet_nlines"][0][2]
    n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
    if n_lines_bulletpoint_start / n_lines > 0.9:
        return False

    # rule 4: fraction of words that contain no alphabetical character must be < 0.4
    no_alph_words = signals["rps_doc_frac_no_alph_words"][0][2]
    if no_alph_words >= 0.4:
        return False

    # rule 5: fraction of stop words in the document must be >= 0.06
    stop_word_fraction = signals["rps_doc_stop_word_fraction"][0][2]
    if stop_word_fraction < 0.06:
        return False

    # rule 6: number of stop words in the document must be >= 2
    stop_word_number = count_stop_words_gopher(line, gopher_stopwords_dir)
    if stop_word_number < 2:
        return False

    # rule 7: ratio of symbols to words in the content must be < 0.5
    symbol_to_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if symbol_to_word_ratio >= 0.5:
        return False

    # rule 8: number of words in the content after normalisation must be between 50, 10000
    word_count = signals["rps_doc_word_count"][0][2]
    if word_count < 50 or word_count > 10_000:
        return False

    return True


def ccnet_rules_pass(signals) -> bool:
    """ function returns True if the sample complies with CCNet rules """

    # rule 1: Head, middle or tail bucket of the perplexity score must <= 0.9
    ccnet_bucket = signals["ccnet_bucket"][0][2]
    if ccnet_bucket > 0.9:
        return False

    # rule 2: Score of the language identification model must be > 0.8
    ccnet_language_score = signals["ccnet_language_score"][0][2]
    if ccnet_language_score < 0.8:
        return False

    # rule 3: Number of characters must be >= 1000
    ccnet_length = signals["ccnet_length"][0][2]
    if ccnet_length < 1000:
        return False

    # rule 4: Number of lines must > 10
    ccnet_nlines = signals["ccnet_nlines"][0][2]
    if ccnet_nlines < 10:
        return False

    # rule 5: Perplexity of an LM trained on Wikipedia must be < 487.5
    ccnet_perplexity = signals["ccnet_perplexity"][0][2]
    if ccnet_perplexity >= 487.5:
        return False

    return True


def c4_rules_pass(signals) -> bool:
    """ function returns True if the sample complies with the filtering rules used in C4 """

    # rule 1: ratio of '{' or '}' must be < 0.025
    curly_bracket = signals["rps_doc_curly_bracket"][0][2]
    if curly_bracket >= 0.025:
        return False

    # rule 2: 'lorem ipsum' must be < 3e-08
    lorem_ipsum = signals["rps_doc_lorem_ipsum"][0][2]
    if lorem_ipsum >= 3 * 10**(-8):
        return False

    # rule 3: number of sentences must be < 7500
    num_sentences = signals["rps_doc_num_sentences"][0][2]
    if num_sentences > 7500:
        return False

    # rule 4: no words contained in the Bad-Words blocklist，这条不采用
    # ldnoobw_words = signals["rps_doc_ldnoobw_words"][0][2]
    # if ldnoobw_words != 0:
        # return False
    return True

def ImpResampling_rules_pass(signals) -> bool:
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """

    # rule 1: the ratio of whether a text is like a book
    books_importance = signals["rps_doc_books_importance"][0][2]
    if books_importance <= -25000 or books_importance >25000:
        return False

    # rule 2: the ratio of whether a text is like a openwebtext must be >-1000
    openwebtext_importancebooks_importance = signals["rps_doc_openwebtext_importance"][0][2]
    if openwebtext_importancebooks_importance < -1000:
        return False

    # rule 3: the ratio of whether a text is like a wikipedia should < 0；暂时不要这条规则
    # wikipedia_importance = signals["rps_doc_wikipedia_importance"][0][2]
    # if wikipedia_importance < 0:
        # return False

    return True


def rpv1_rules_pass(signals) -> bool:
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """

    # rule 1: the wikipedia reference classifier score must be higher than 0.025
    wikiref_score = signals["rps_doc_ml_wikiref_score"][0][2]
    if wikiref_score < 0.025:
        return False

    # rule 2: the palm score must be 这条规则不采用
    # palm_score = signals["rps_doc_ml_palm_score"][0][2]
    # if palm_score <= 0.1:
        # return False

    return True

def pretrainer_rules_pass(signals) -> bool:
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """

    # rule 1: fraction of words in the content that only consist of uppercase letters must be < 0.1
    all_caps_words = signals["rps_doc_frac_all_caps_words"][0][2]
    if all_caps_words >= 0.1:
        return False

    # rule 2: fraction of unique words must be within [0.2, 0.8]
    unique_words = signals["rps_doc_frac_unique_words"][0][2]
    if unique_words < 0.2 or unique_words > 0.8:
        return False

    return True


def refinedweb_rules_pass(signals):
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """

    # rule 1: entropy of the unigram distribution of the content must be within [3, 6]
    unigram_entropy = signals["rps_doc_unigram_entropy"][0][2]
    if unigram_entropy < 3 or unigram_entropy > 6:
        return False

    # rule 2: the ratio between characters in the most frequent 2-gram and the total number of characters must be below 0.2
    top_2_gram_frac = signals["rps_doc_frac_chars_top_2gram"][0][2]
    if top_2_gram_frac > 0.2:
        return False

    # rule 3: the ratio between characters in the most frequent 3-gram and the total number of characters must be below 0.18
    top_3_gram_frac = signals["rps_doc_frac_chars_top_3gram"][0][2]
    if top_3_gram_frac > 0.18:
        return False

    # rule 4: the ratio between characters in the most frequent 4-gram and the total number of characters must be below 0.16
    top_4_gram_frac = signals["rps_doc_frac_chars_top_4gram"][0][2]
    if top_4_gram_frac > 0.16:
        return False

    # rule 5: the ratio between characters in the duplicate 5-gram and the total number of characters must be below 0.15
    dupe_5_gram_frac = signals["rps_doc_frac_chars_dupe_5grams"][0][2]
    if dupe_5_gram_frac > 0.15:
        return False

    # rule 6: the ratio between characters in the duplicate 6-gram and the total number of characters must be below 0.14
    dupe_6_gram_frac = signals["rps_doc_frac_chars_dupe_6grams"][0][2]
    if dupe_6_gram_frac > 0.14:
        return False

    # rule 7: the ratio between characters in the duplicate 7-gram and the total number of characters must be below 0.13
    dupe_7_gram_frac = signals["rps_doc_frac_chars_dupe_7grams"][0][2]
    if dupe_7_gram_frac > 0.13:
        return False

    # rule 8: the ratio between characters in the duplicate 8-gram and the total number of characters must be below 0.12
    dupe_8_gram_frac = signals["rps_doc_frac_chars_dupe_8grams"][0][2]
    if dupe_8_gram_frac > 0.12:
        return False

    # rule 9: the ratio between characters in the duplicate 9-gram and the total number of characters must be below 0.11
    dupe_9_gram_frac = signals["rps_doc_frac_chars_dupe_9grams"][0][2]
    if dupe_9_gram_frac > 0.11:
        return False

    # rule 10: the ratio between characters in the duplicate 10-gram and the total number of characters must be below 0.10
    dupe_10_gram_frac = signals["rps_doc_frac_chars_dupe_10grams"][0][2]
    if dupe_10_gram_frac > 0.10:
        return False

    # rule 11: No urls in blacklist should exist in the text
    ut1_blacklist = signals["rps_doc_ut1_blacklist"][0][2]
    if ut1_blacklist != 0 and ut1_blacklist:
        return False

    # we 
    # # rule 12 & 13: ratio of duplicated sentences and characters in dupe sents must be < 0.30 and < 0.20
    # num_sentences = signals["rps_doc_num_sentences"][0][2]
    # num_characters = signals["ccnet_length"][0][2]
    # duplicated_sents, duplicated_chars = count_duplicates(line1)
    # if duplicated_sents / num_sentences > 0.30 or duplicated_chars / num_characters > 0.20:
    #     return False

    return True

def self_defined_rules(line, signals):
    """ function returns True if the sample complies with self-defined filtering rules """
    sentences = sent_tokenize(line)
    words = word_tokenize(line)
    word_count = signals["rps_doc_word_count"][0][2]

    # rule 1: Mean word length must be between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # rule 2: Sentences contained in the whole document must be > 1
    if len(sentences) == 1:
        return False

    # rule 3: The fraction of lines end with 'readmore' must be <= 0.1'
    readmore_count = 0
    for sentence in sentences:
        if re.match(r'.*(read more)$', sentence.lower()):
            readmore_count += 1
    if readmore_count / len(sentences) > 0.1:
        return False

    # rule 4: The fraction of nonconsecutive hashtags in words must be <= 0.1
    hash_count = line.count('#')
    if hash_count >= 1:
        consecutive_hashes_matches = re.findall(r'#+', line)
        consecutive_hashes = sum(len(match) for match in consecutive_hashes_matches)
        nonconsecutive_hashes = hash_count - consecutive_hashes
        nonconsecutive_hash_ratio = nonconsecutive_hashes / word_count
        if nonconsecutive_hash_ratio > 0.1:
            return False

    # rule 5: The fraction of nonconsecutive ellipsis in words must be <= 0.1
    ellipsis = ['\\.{3}', '……', '…']  # 对于正则表达式，需要使用双反斜杠转义点号
    ellipsis_count = sum(line.count(symbol) for symbol in ellipsis)
    if ellipsis_count >= 1:
        # 最后一种省略号连续出现三个以上才算连续出现
        ellipsis_pattern = '|'.join(re.escape(sym) + '{2,}' if sym != '…' else re.escape(sym) + '{3,}' for sym in ellipsis)
        consecutive_ellipses_matches = re.findall(f'({ellipsis_pattern})+', line)
        # 计算连续省略号的次数
        consecutive_ellipses = sum(len(match) // 3 for match in consecutive_ellipses_matches)  # 一个省略号包含三个点
        nonconsecutive_ellipses = ellipsis_count - consecutive_ellipses
        nonconsecutive_ellipsis_ratio = nonconsecutive_ellipses / word_count
        if nonconsecutive_ellipsis_ratio > 0.1:
            return False

    # rule 6: The fraction of punctuations in words must > 0
    punct_count = sum(1 for char in line if char in string.punctuation)
    if punct_count == 0:
        return False

    # rule 7: The fraction of non-alpha words over non-punctuation words must be below 0.2
    '''
    # 使用 'words' 数据集判断是否为英文单词
    non_english_words_count = 0
    english_vocab = set(nltk.corpus.words.words())
    english_vocab = EC.Dict("en_US")
    for word in words:
        if not english_vocab.check(word.lower()):
            non_english_words_count += 1
    '''
    non_english_words_count = sum(1 for word in words if not word.isalpha())
    non_punct_words_count = len(re.sub(r'[^\w\s]', '', line))
    if non_english_words_count / non_punct_words_count > 0.2:
        return False

    # rule 8: The fraction of digital words over non-punctuation words must be below 0.3
    # 计算非数字词的数量
    digit_words_count = sum(1 for word in words if word.isdigit())
    if digit_words_count / non_punct_words_count > 0.3:
        return False

    # rule 9: 'qs_url_cate_bad_words': 'lambda x: x != 0'
    for word in words:
        if word in STRICT_BAD_URL_WORDS or word in HARD_BAD_URL_WORDS:
            return False

    return True




def count_stop_words_gopher(line, filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.readlines()[0]
        stop_words = json.loads(text)
        # stop_words = [line.strip() for line in file]
    count = 0
    # 对文本去除标点符号，保留stop_words中出现的标点符号\和'
    line = re.sub(r'[^\w\s\'\\]', '', line)
    words = word_tokenize(line.lower())
    for word in words:
        if word in stop_words:
            count += 1
            stop_words.remove(word)   # 计算停用词在文本中出现的个数而非次数
            if count >= 2:  # 如果发现停用词数量已经超过两个，直接break
                break
    return count


# 计算重复句子和字符
def count_duplicates(line):
    sentences = sent_tokenize(line['raw_content'])
    duplicated_sents = 0
    duplicated_chars = 0

    # 对句子进行重复率检测
    sent_similarity_matrix = calculate_similarity_matrix(line, sentences)
    for i in range(len(sentences)):
        similarity_scores = sent_similarity_matrix[i, i + 1:]
        # 句子相似度超过0.9判断为两个句子重复
        similar_indices = np.where(similarity_scores > 0.9)[0] + i + 1
        for j in similar_indices:
            duplicated_sents += 1
            duplicated_chars += len(sentences[i])
    return duplicated_sents, duplicated_chars


# 计算重复字段的相似度矩阵
def calculate_similarity_matrix(line, sentences):
    try:
        # 加载 TF-IDF 值
        tfidf_dict = line["tf_idf"]
        # 提取所有句子的词汇表
        vocab = set(tfidf_dict.keys()).union(*[set(sentence.split()) for sentence in sentences])
        # 使用 TfidfVectorizer 构建整篇文本的 TF-IDF 矩阵
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return similarity_matrix
    except ValueError as e:
        # 捕获 ValueError，检查是否是空词汇表导致的错误
        if "empty vocabulary" in str(e):
            # print("Empty vocabulary detected. Exiting.")
            return 0
        else:
            # 如果不是空词汇表导致的错误，将异常重新引发
            raise
    except FileNotFoundError as e:
        print(f"Text not found: {line}")
        return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0


def filter_dataset(args):
    with open(args.input_data_dir, 'r') as data_file, open(args.input_signals_dir, 'r') as signals_file:
        texts = data_file.readlines()
        signals = signals_file.readlines()
        for index, line in enumerate(texts):
            line1 = json.loads(line)
            line = line1['raw_content']
            signal = json.loads(signals[index])["quality_signals"]
            if not gopher_rules_pass(line, signal):
                # 判断为需要过滤
                dump_or_not = True
                continue
            if not ccnet_rules_pass(signal):
                dump_or_not = True
                continue
            if not c4_rules_pass(signal):
                dump_or_not = True
                continue
            if not ImpResampling_rules_pass(signal):
                dump_or_not = True
                continue
            if not rpv1_rules_pass(signal):
                dump_or_not = True
                continue
            if not pretrainer_rules_pass(signal):
                dump_or_not = True
                continue
            if not refinedweb_rules_pass(line1, signal, args):
                dump_or_not = True
                continue
            if not self_defined_rules(line, signal):
                dump_or_not = True
                continue
            # 合格，保留当前字段
            dump_or_not = False

def filter_func(data):
    try:
        signal = data
        line = data['raw_content']
        if not gopher_rules_pass(line, signal):
            # 判断为需要过滤
            return False
        if not ccnet_rules_pass(signal):
            return False
        if not c4_rules_pass(signal):
            return False
        if not ImpResampling_rules_pass(signal):
            return False
        if not rpv1_rules_pass(signal):
            return False
        if not pretrainer_rules_pass(signal):
            return False
        if not refinedweb_rules_pass(signal):
            return False
        if not self_defined_rules(line, signal):
            return False
        # 合格，保留当前字段
        return True
    except Exception as e:
        # 当上述代码块发生任何异常时执行
        print("error is:", e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", help="Dataset input directory.")
    parser.add_argument("--input_signals_dir", help="Corresponding quality siganals file input directory.")
    # parser.add_argument("--output_file_final", help="File to output final docs to.")
    parser.add_argument("--gopher_stopwords_dir", help="Gopher stop words file path.")
    args = parser.parse_args()
    filter_dataset(args)