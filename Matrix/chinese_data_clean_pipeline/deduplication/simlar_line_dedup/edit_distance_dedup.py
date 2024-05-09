from collections import Counter

def minDistance(word1: str, word2: str, threshold) -> bool:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        # 第一行
        for j in range(1, n2 + 1):
            dp[0][j] = dp[0][j-1] + 1
        # 第一列
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i-1][0] + 1
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] ) + 1
                    # if dp[i][j] > threshold:
                    #     return False
        #print(dp)      
        return dp[-1][-1] <= threshold

def split_text(text):
    sentence_split_sign = ['[', '.', '!', '?', '。', '！', '？', '\\', '……', '…', '【', ']', '】']
    begin_offset = 0
    sentence_list = []
    for i in range(len(text)):
        if text[i] in sentence_split_sign:
            if text[i] not in ['[', '【']:
                sentence_list.append(text[begin_offset:i+1])
                begin_offset = i+1
            else:
                if begin_offset < i:
                    sentence_list.append(text[begin_offset:i])
                begin_offset = i
    if begin_offset < len(text):
        sentence_list.append(text[begin_offset:])
    return sentence_list

def dedup_text(text, proportion):
    sentences = split_text(text)
    remove_sentence_indx = []
    for i in range(len(sentences)):
        if i in remove_sentence_indx:
            continue
        for j in range(i+1, len(sentences)):
            if abs(len(sentences[i]) - len(sentences[j])) / max(len(sentences[i]), len(sentences[j])) < max(0.3, proportion):
                if compute_simlaritry_word_freq(sentences[i], sentences[j]) < 1/3:
                    continue
                threshold = proportion * max(len(sentences[i]), len(sentences[j]))
                if min(len(sentences[i]), len(sentences[j])) <= 15:
                    threshold = 0
                flag = minDistance(sentences[i], sentences[j], threshold)
                if flag:
                    remove_sentence_indx.append(j)

    remain_sentence = [sentences[i] for i in range(len(sentences)) if i not in remove_sentence_indx]
    filtered_article = "".join(remain_sentence)
    return filtered_article

def compute_simlaritry_word_freq(sent1, sent2):
    # words1 = jieba.cut(sent1)
    word_counter1 = Counter(sent1)
    # words2 = jieba.cut(sent2)
    word_counter2 = Counter(sent2)
    union_words = 0
    intersection_words = 0
    for word1 in word_counter1:
        if word1 in word_counter2:
            intersection_words += min(word_counter1[word1], word_counter2[word1])
            union_words += max(word_counter1[word1], word_counter2[word1])
        else:
            union_words += word_counter1[word1]
    for word2 in word_counter2:
        if word2 not in word_counter1:
            union_words += word_counter2[word2]
    return intersection_words / union_words

def dedup_text_by_word_freq(text, proportion):
    sentences = split_text(text)
    remove_sentence_indx = []
    for i in range(len(sentences)):
        if i in remove_sentence_indx:
            continue
        for j in range(i+1, len(sentences)):
            similarity = compute_simlaritry_word_freq(sentences[i], sentences[j])
            
            if similarity > proportion:
                remove_sentence_indx.append(j)
    remain_sentence = [sentences[i] for i in range(len(sentences)) if i not in remove_sentence_indx]
    filtered_article = "".join(remain_sentence)
    return filtered_article 

if __name__ == '__main__':
    # 示例中文文章
    article = """在日军的重重封锁之下,《行路难》是此时的中国广大人民最适合的诠释.在江西和浙江,中日军队再次激战,战役结束后,浙赣两省的机场被彻底破坏,20余万居民及士兵惨遭日军杀害.这一年,八路军、新四军由五十万人减为四十万人,抗日根据地面积缩小.此时的日本国内也并不好过,紧张的经济使得日本人民均承受着政府繁重的压力.日军切断滇缅公路后,新开辟的驼峰航线成了国内唯一一条运输通路,美军在这条航路上为中国提供了大量援助.在日军的重重封锁之下,《行路难》是此时的中国广大人民最适合的诠释.8月份,浙赣战役结束后,浙赣两省的机场被彻底破坏,20余万居民及士兵惨遭日军杀害.这一年,八路军、新四军由五十万人减为四十万人,抗日根据地面积缩小.此时的日本国内并不好过,紧张的经济使得日本人民均承受着政府的压力,从学生到妇女都做着繁重的工作.在四川,盲肠炎和疟疾成了最为流行的疾病.日军切断滇缅公路后,新开辟的驼峰航线成了国内唯一一条运输通路,美军在这条航路上为中国提供了大量援助.苏联此时已是白雪皑皑,11月19日,苏联军队开始展开反攻.两个月后,苏军全歼被围德军,斯大林格勒保卫战成为二战的转折点.在日军的重重封锁之下,《行路难》是此时的中国广大人民最适合的诠释.在江西和浙江,中日军队再次激战,战役结束后,浙赣两省的机场被彻底破坏,20余万居民及士兵惨遭日军杀害.这一年,八路军、新四军由五十万人减为四十万人,抗日根据地面积缩小.此时的日本国内也并不好过,紧张的经济使得日本人民均承受着政府繁重的压力.日军切断滇缅公路后,新开辟的驼峰航线成了国内唯一一条运输通路,美军在这条航路上为中国提供了大量援助."""
    print(dedup_text(article, 0.4))
    print(dedup_text_by_word_freq(article, 0.6))