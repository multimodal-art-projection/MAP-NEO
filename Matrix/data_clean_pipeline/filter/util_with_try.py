import argparse
import unicodedata
import numpy as np
import string
import time
from bad_url_words import STRICT_BAD_URL_WORDS, HARD_BAD_URL_WORDS
from nltk import sent_tokenize, word_tokenize
from nltk.util import ngrams
import nltk.corpus
from collections import defaultdict, Counter
import re
import json
import fasttext
import math
import os
import fileinput
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

parser = argparse.ArgumentParser()
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

strict_bad_url_word = set()
for word in STRICT_BAD_URL_WORDS:
    strict_bad_url_word.add(word)
hard_bad_url_word = set()
for word in HARD_BAD_URL_WORDS:
    hard_bad_url_word.add(word)

model = fasttext.load_model(args.fasttext_model_dir)

def gopher_rules_pass(text, args):
    """ function returns True if the sample complies with Gopher rules """
    total_character_count = len(text)
    if total_character_count == 0:
        return False, {'total character count': total_character_count}
    normalized_content = unicodedata.normalize('NFKC', text)
    sentences = re.split(r'[.!…?;"]+', normalized_content)  # Using regular expression to split the document into sentences
    words = word_tokenize(normalized_content)
    word_count = len(words)

    # rule 1: mean word length between 3 and 10
    char_count = 0
    for w in words:
        char_count += len(w)
    mean_word_length = char_count / word_count if word_count else 0
    if mean_word_length < args.threshold_rps_doc_mean_word_length_1 or mean_word_length > args.threshold_rps_doc_mean_word_length_2:
        return False, {'rps_doc_mean_word_length': mean_word_length}

    # rule 2: fraction of lines that end with an ellipsis must be below 0.2
    ellipsis_pattern1 = re.compile(r'…+')
    ellipsis_pattern2 = re.compile(r'\.{2,}')
    # 计算匹配到的次数
    ellipsis_count = len(re.findall(ellipsis_pattern1, text)) + len(re.findall(ellipsis_pattern2, text))
    frac_ellipsis = ellipsis_count / len(text) if text else 1
    if frac_ellipsis > args.threshold_rps_doc_frac_lines_end_with_ellipsis:
        return False, {'rps_doc_frac_lines_end_with_ellipsis': frac_ellipsis}

    # rule 3: 90% of lines need to start without a bullet point
    # Split the content into lines
    lines = text.splitlines()
    # Define common bullet point characters
    bullet_points = ['•', '●', '○', '■', '□', '▪', '▫', '※', '·', '➢']
    utf_code = ['\u2022', '\u2023', '\u25B6', '\u25C0', '\u25E6', '\u25A0', '\u25A1', '\u25AA', '\u25AB', '\u2013']
    bullet_points += utf_code
    # Count the number of lines starting with a bullet point
    bullet_point_count = sum(
        1 for line in lines if line and any(line.startswith(bullet) for bullet in bullet_points))
    # Calculate the fraction of lines starting with a bullet point
    fraction = bullet_point_count / len(lines) if lines else 0
    # Check if the fraction is less than or equal to 0.9
    if fraction > args.threshold_rps_lines_start_with_bulletpoint:
        return False, {'rps_lines_start_with_bulletpoint': fraction}

    # rule 4: fraction of words that contain no alphabetical character must be < 0.4
    text1 = text.strip()
    text1 = text1.split()
    cnt = 0
    for word in text1:
        word = list(word)
        no_alpha_flag = True
        for c in word:
            if c.isalpha():
                no_alpha_flag = False
                break
        cnt += 1 if no_alpha_flag else 0
    no_alpha_frac = cnt / len(text)
    if no_alpha_frac >= args.no_alph_words_th:
        return False, {'no_alpha_frac': no_alpha_frac}

    # rule 5: fraction of stop words in the document must be >= 0.06
    nopunct_word_count, stop_word_number = count_stop_words_gopher(text, args.gopher_stopwords_dir)
    # if stop_word_number/nopunct_word_count < args.frac_stop_words_gopher:
        # return False, {'stop_word_ratio': stop_word_number/nopunct_word_count}

    # rule 6: number of stop words in the document must be >= 2
    if stop_word_number < 2:
        return False, {'stop_word_count': stop_word_number}

    # rule 7: ratio of symbols to words in the content must be < 0.5
    symbols = ["#", "...", "…"]
    symbol_count = 0
    for symbol in symbols:
        symbol_count += text.count(symbol)
    if symbol_count / word_count >= args.frac_symbol_words:
        return False, {'symbol to words ratio': symbol_count / word_count}

    # rule 8: number of words in the content after normalisation must be between 50, 10000
    if word_count > args.max_word_count or word_count < args.min_word_count:
        return False, {'Word count out of range': word_count}

    return True, None


def ccnet_rules_pass(text, args):
    """ function returns True if the sample complies with CCNet rules """

    # rule 1: Score of the language identification model must be > 0.4
    text1 = text.replace('\n', '')
    predict = model.predict(text1, k=1)
    score = predict[1][0]
    if score < args.threshold_fasttext_score:
        return False, {'fasttext score': score}

    # rule 2: Number of characters must be >= 200
    test_line = text.strip()
    test_line = test_line.replace(" ", "")
    test_line = test_line.replace("\n", "")
    test_line = test_line.replace("\t", "")
    length = len(test_line)
    if length < args.ccnt_min_length_th:
        return False, {'min_num_characters': length}

    # rule 3: Number of lines must > 1
    lines = text.split('\n')
    if len(lines) < args.theshold_num_lines:
        return False, {'number of lines': len(lines)}

    return True, None


def c4_rules_pass(line, args):
    """ function returns True if the sample complies with the filtering rules used in C4 """
    normalized_content = unicodedata.normalize('NFKC', line)

    # rule 1: ratio of '{' or '}' must be < 0.025
    curly_braces_count = line.count('{') + line.count('}')
    ratio_curly_braces = curly_braces_count / len(line)
    if ratio_curly_braces >= args.threshold_curly_braces:
        return False, {'curly_braces_ratio': ratio_curly_braces}

    # rule 2: 'lorem ipsum' must be < 3e-08
    lorem_ipsum_count = normalized_content.lower().count('lorem ipsum')
    ratio_lorem_ipsum = lorem_ipsum_count / len(normalized_content)
    if ratio_lorem_ipsum >= args.threshold_lorem_ipsum:
        return False, {'lorem_ipsum_ratio': ratio_lorem_ipsum}

    # rule 3: number of sentences must be < 7500
    sentence_count = len(re.findall(r'\b[^.!?]+[.!?]*', normalized_content))
    if sentence_count >= args.threshold_sentence_count:
        return False, {'sentence_count': sentence_count}

    return True, None



def pretrainer_rules_pass(line):
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """

    # rule 1: fraction of words in the content that only consist of uppercase letters must be < 0.1
    words = line.split()
    uppercase_word_count = 0
    total_word_count = 0
    for word in words:
        if word.isalpha() and word.isupper():
            uppercase_word_count += 1
        total_word_count += 1
    rps_doc_frac_all_caps_words = uppercase_word_count / total_word_count if total_word_count > 0 else 0
    if rps_doc_frac_all_caps_words > args.threshold_frac_all_caps_words:
        return False, {'rps_doc_frac_all_caps_words': rps_doc_frac_all_caps_words}

    # rule 2: fraction of unique words must be within [0.1, +inf)
    unique_words = set(words)
    rps_doc_frac_unique_words = len(unique_words) / len(words) if len(words) > 0 else 0
    if rps_doc_frac_unique_words < args.lower_threshold_frac_unique_words:
        return False, {'rps_doc_frac_unique_words': rps_doc_frac_unique_words}

    return True, None


def refinedweb_rules_pass(text, args):
    """ function returns True if the sample complies with the filtering rules used in RP-V1 """
    total_character_count = len(text)
    if total_character_count == 0:
        return False, {'total character count': total_character_count}

    # rule 1: entropy of the unigram distribution of the content must be within [3, 6]
    words = text.split()
    unigram_counts = Counter(words)
    total_unigrams = sum(unigram_counts.values())
    # Calculate the entropy of the unigram distribution
    rps_doc_unigram_entropy = -sum(
        (count / total_unigrams) * math.log(count / total_unigrams) for count in unigram_counts.values())
    if rps_doc_unigram_entropy < args.lower_threshold_doc_unigram_entropy:
        return False, {'rps_doc_unigram_entropy': rps_doc_unigram_entropy}

    # rule 2: the ratio between characters in the most frequent 2-gram and the total number of characters must be below 0.2
    top2gram_char = count_ngram_duplicates(text, 2)
    if top2gram_char / total_character_count > args.threshold_frac_chars_top2gram:
        return False, {'characters in the top word 2gram': top2gram_char / total_character_count}

    # rule 3: the ratio between characters in the most frequent 3-gram and the total number of characters must be below 0.18
    top3gram_char = count_ngram_duplicates(text, 3)
    if top3gram_char / total_character_count > args.threshold_frac_chars_top3gram:
        return False, {'characters in the top word 3gram': top3gram_char / total_character_count}

    # rule 4: the ratio between characters in the most frequent 4-gram and the total number of characters must be below 0.16
    top4gram_char = count_ngram_duplicates(text, 4)
    if top4gram_char / total_character_count > args.threshold_frac_chars_top4gram:
        return False, {'characters in the top word 4gram': top4gram_char / total_character_count}

    # rule 5: the ratio between characters in the duplicate 5-gram and the total number of characters must be below 0.15
    dupe5gram_char = count_ngram_duplicates(text, 5)
    if dupe5gram_char / total_character_count > args.threshold_frac_chars_dupe5gram:
        return False, {'characters in the duplicated 5gram': dupe5gram_char / total_character_count}

    # rule 6: the ratio between characters in the duplicate 6-gram and the total number of characters must be below 0.14
    dupe6gram_char = count_ngram_duplicates(text, 6)
    if dupe6gram_char / total_character_count > args.threshold_frac_chars_dupe6gram:
        return False, {'characters in the duplicated 6gram': dupe6gram_char / total_character_count}

    # rule 7: the ratio between characters in the duplicate 7-gram and the total number of characters must be below 0.13
    dupe7gram_char = count_ngram_duplicates(text, 7)
    if dupe7gram_char / total_character_count > args.threshold_frac_chars_dupe7gram:
        return False, {'characters in the duplicated 7gram': dupe7gram_char / total_character_count}

    # rule 8: the ratio between characters in the duplicate 8-gram and the total number of characters must be below 0.12
    dupe8gram_char = count_ngram_duplicates(text, 8)
    if dupe8gram_char / total_character_count > args.threshold_frac_chars_dupe8gram:
        return False, {'characters in the duplicated 8gram': dupe8gram_char / total_character_count}

    # rule 9: the ratio between characters in the duplicate 9-gram and the total number of characters must be below 0.11
    dupe9gram_char = count_ngram_duplicates(text, 9)
    if dupe9gram_char / total_character_count > args.threshold_frac_chars_dupe9gram:
        return False, {'characters in the duplicated 9gram': dupe9gram_char / total_character_count}

    # rule 10: the ratio between characters in the duplicate 10-gram and the total number of characters must be below 0.10
    dupe10gram_char = count_ngram_duplicates(text, 10)
    if dupe10gram_char / total_character_count > args.threshold_frac_chars_dupe10gram:
        return False, {'characters in the duplicated 10gram': dupe10gram_char / total_character_count}

    # # rule 11: No urls in blacklist should exist in the text
    # files = os.listdir(args.ut1_url_blacklist_dir)  # 遍历文件路径，输出目录下的文件名和文件夹名
    # # check if the html page is related to forbidden urls
    # for f in files:
    #     file_name = args.ut1_url_blacklist_dir + '/' + f + '/' + 'urls'
    #     try:
    #         with open(file_name, 'r') as fp:
    #             for forbidden_url in fp.readlines():
    #                 forbidden_url = forbidden_url.replace('\n', '')
    #                 # if forbidden_url in line['url'] or forbidden_url in line['text']:
    #                 if forbidden_url in text:
    #                     return False, None, {'forbidden url': forbidden_url}
    #     except FileNotFoundError as e:
    #         continue
    #     except NotADirectoryError as e:
    #         continue

    # # 需要提前save tf-idf value的规则
    # # rule 12 & 13: ratio of duplicated sentences and characters in dupe sents must be < 0.30 and < 0.20
    # duplicated_sents_ratio, duplicated_chars = count_duplicates(line)
    # if duplicated_sents_ratio > args.threshold_frac_lines_dupe_lines:
    #     return False, {'duplicated sentences ratio': duplicated_sents_ratio}
    # if duplicated_chars / total_character_count > args.threshold_frac_chars_dupe_lines:
    #     return False, {'duplicated sentences\' chars ratio': duplicated_chars / total_character_count}

    return True, None

def self_defined_rules(line, args):
    """ function returns True if the sample complies with self-defined filtering rules """
    sentences = sent_tokenize(line)
    words = word_tokenize(line)

    # rule 1: Sentences contained in the whole document must be > 1
    if len(sentences) <= args.threshold_num_sents:
        return False, {'number_of_senteces': len(sentences)}

    # rule 2: The fraction of lines end with 'readmore' must be <= 0.1'
    readmore_count = 0
    for sentence in sentences:
        if re.match(r'.*(read more)$', sentence.lower()):
            readmore_count += 1
    if readmore_count / len(sentences) > args.frac_lines_endwith_readmore:
        return False, {'readmore_fraction': readmore_count / len(sentences)}

    # rule 3: The fraction of nonconsecutive hashtags in words must be <= 0.1
    hashes_count = len(re.findall(r'#+', line))
    nonconsecutive_hash_ratio = hashes_count / len(line)
    if nonconsecutive_hash_ratio > args.nonconsecutive_hash_ratio:
        return False, {'hash_fraction': nonconsecutive_hash_ratio}

    # rule 4: The fraction of nonconsecutive ellipsis in words must be <= 0.1
    # 使用正则表达式匹配省略号和句号
    ellipsis_pattern1 = re.compile(r'…+')
    ellipsis_pattern2 = re.compile(r'\.{2,}')
    # 计算匹配到的次数
    ellipsis_count = len(re.findall(ellipsis_pattern1, line)) + len(re.findall(ellipsis_pattern2, line))
    if ellipsis_count / len(line) > args.nonconsecutive_ellipsis_ratio:
        return False, {'ellipsis_fraction': ellipsis_count / len(line)}

    # rule 5: The fraction of punctuations in words must > 0
    punct_count = sum(1 for char in line if char in string.punctuation)
    if punct_count <= args.threshold_frac_punct:
        return False, {'punct_count': punct_count}

    # rule 6: The fraction of non-alpha words over non-punctuation words must be below 0.2
    non_english_words_count = sum(1 for word in words if not word.isalpha())
    non_punct_words_count = len(re.sub(r'[^\w\s]', '', line))
    if non_english_words_count / non_punct_words_count > args.frac_words_no_alph_over_no_punct:
        return False, {'non_english_words_fraction': non_english_words_count / non_punct_words_count}

    # rule 7: The fraction of digital words over non-punctuation words must be below 0.3
    # 计算非数字词的数量
    digit_words_count = sum(1 for word in words if word.isdigit())
    if digit_words_count / non_punct_words_count > args.frac_words_digital_over_no_punct:
        return False, {'digit_word_fraction': digit_words_count / non_punct_words_count}

    # rule 8: No bad words should be contained in the text
    for word in words:
        if word in strict_bad_url_word or word in hard_bad_url_word:
            return False, {'word in bad words list': word}

    return True, None


def count_stop_words_gopher(line, filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.readlines()[0]
        stop_words = json.loads(text)
        # stop_words = [line.strip() for line in file]
    count = 0
    # 对文本去除标点符号，保留stop_words中出现的标点符号\和'
    line = re.sub(r'[^\w\s\'\\]', '', line)
    words = line.lower().split()
    for word in words:
        if word in stop_words:
            count += 1
            stop_words.remove(word)   # 计算停用词在文本中出现的个数而非次数
    return len(words), count

def count_ngram_duplicates(text, n):
    # 进行重复ngram的判断
    ngram_counts = {}  # 用于存储𝑛-gram及其出现次数的字典
    # 计算n-grams及其出现次数
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    # 对n-grams按出现次数降序排序
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    # 对于[2,4]范围内的n，我们计算出现次数最多的n-gram中包含的字符所占的比例
    if sorted_ngrams and sorted_ngrams[0]:
        if n <= 4:
            # 只处理最常出现的 n-gram
            ngram = sorted_ngrams[0][0]
            ngram_counts[ngram] -= 1  # 避免计算与自身的重复次数
            return ngram_counts[ngram] * len(ngram)
        # 对于[5,10]范围内的n，我们计算所有重复的n元词串中包含的字符的比例
        else:
            duplicate_ngrams = set()  # 用于存储所有重复的n-grams
            overlapping_characters = set()  # 用于存储重叠n-grams中的字符
            for ngram, count in sorted_ngrams:
                if count > 1:
                    duplicate_ngrams.add(ngram)
                    overlapping_characters.update(set(ngram))
            return len(overlapping_characters)
    return 0

# 计算重复句子和字符
def count_duplicates(line):
    sentences = sent_tokenize(line['text'])
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
    return duplicated_sents / len(sentences), duplicated_chars


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



def filter_one_line_wrap(args):
    def filter_one_line(text):
        try:
            retain_or_not, flag = gopher_rules_pass(text, args)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'gopher_rules_pass'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        try:
            retain_or_not, flag = ccnet_rules_pass(text, args)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'ccnet_rules_pass'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        try:
            retain_or_not, flag = c4_rules_pass(text, args)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'c4_rules_pass'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        try:
            retain_or_not, flag = pretrainer_rules_pass(text)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'pretrainer_rules_pass'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        try:
            retain_or_not, flag = self_defined_rules(text, args)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'self_defined_rules'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        try:
            retain_or_not, flag = refinedweb_rules_pass(text, args)
            if not retain_or_not:
                print(f"filter reason: {flag}")
                return False
        except Exception as e:
            rule = 'refinedweb_rules_pass'
            print(f"rule: {rule}, Exception happen: {e}")
            return False

        return True

    return filter_one_line