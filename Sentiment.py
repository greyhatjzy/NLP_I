import os
import pandas as pd
import jieba
import jieba.posseg as pseg
import re

pd.set_option("display.max_columns", 50)
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def jiebafenci(text):
    re = ""
    words = pseg.cut(text)
    for w in words:
        flag = w.flag
        tmp = w.word
        if len(tmp) > 1 and len(flag) > 0:
            re = re + " " + w.word

    re = re.replace("\n", " ").replace("\r", " ")
    return re


def tokenization_s(sentences):  # same can be achieved for words tokens
    s_new = []
    for sent in (sentences):  # For NumpY = sentences[:]
        print('sent',sent)
        s_token = sent_tokenize(sent)
        print('s_token',s_token)
        if s_token != '':
            s_new.append(s_token)
        break
    return s_new


def tokenization_w(words):
    w_new = []
    # print(words)
    for w in (words):  # for NumPy = words[:]
        print('w',w)
        try:
            w_token = word_tokenize(w)
        except:
            pass
            # print('error ',w)
        try:
            if w_token != '':
                w_new.append(w_token)
        except:
            pass
        # break
    return w_new


def preprocess(text):
    clean_data = []
    for x in (text[:][0]):
        new_text = re.sub('<.*?>', '', x)  # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punc.
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = new_text.lower()  # lower case, .upper() for upper
        if new_text != '':
            clean_data.append(new_text)
    return clean_data


if __name__ == '__main__':
    os.chdir('/home/jzy/Desktop/NLP_I')

    # # 处理原始数据，词汇编码
    original_data = pd.read_csv('./data/simplifyweibo_4_moods.csv')
    #  分词
    # original_data['review_fenci'] = original_data['review'].apply(jiebafenci)

    # original_data.to_csv('simplifyweibo_4_moods_fenci.csv', encoding='gb18030',index=False)
    # print(original_data.head())
    # print(len(original_data['label']))

    original_data = pd.read_csv('./simplifyweibo_4_moods_fenci.csv',
                                encoding='gb18030', dtype={'review_fenci': 'str'})

    # print(original_data['review_fenci'].head(10))
    # print(original_data.head())

    # data_label = original_data['label']
    # data_fenci = original_data['review_fenci'].str.split(' ', expand=True)
    # data_fenci.to_csv('simplifyweibo_4_moods_fenci_expand.csv', encoding='gb18030', index=False)

    # print(data_fenci.head())

    # 建立token词典
    # 比较nltk和TF之间的区别
    # 方法一 nltk：
    # nltk 的分词有问题，分词不起作用
    # reference:
    # https://medium.com/biaslyai/beginners-guide-to-text-preprocessing-in-python-2cbeafbf5f44
    # https://stackoverflow.com/questions/33098040/how-to-use-word-tokenize-in-data-frame

    # sent_tokenize_list = word_tokenize(original_data['review_fenci'])

    sentences = tokenization_s(original_data['review_fenci'])
    sent_tokenize_list = tokenization_w(sentences)
    print('sent_tokenize_list',sent_tokenize_list)

    # 方法二 TF：
    # token = Tokenizer(num_words=2000)
    # token.fit_on_texts(original_data['review_fenci'])
    # print(token.document_count)
    # print(token.word_index)
    # simplifyweibo_4_moods一共361744条，取80%为tain
