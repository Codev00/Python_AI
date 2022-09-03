from gettext import npgettext
from operator import imod
import nltk
import pickle
import numpy as np 

from keras.models import load_model
model = load_model('train_model.h5')
import json
import random

intents = json.loads(open('./intents.json'), encoding='utf8').read()
words= pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))

# Tiền xử lý văn bản - ý định của người dùng
# Chia câu thành tokenizer
def tokenizer_sentence(sentence):
    sentence_w = nltk.word_tokenize(sentence)
    sentence_w = [ w.lower() for w in sentence_w ]
    return sentence_w

# Hàm tìm phần tử giống với dữ liệu train
def bag_element(sentence, words):
    sentence_w = tokenizer_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_w:
        for i,w in enumerate(words): #enumarate sinh index bắt đầu từ 0
            if w == s:
                bag[i] = 1
    return (np.array(bag))