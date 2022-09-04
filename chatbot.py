from audioop import reverse
from gettext import npgettext
from operator import imod
from unittest import result
import nltk
import pickle
import numpy as np 

from keras.models import load_model
model = load_model('train_model.h5')
import json
import random

intents = json.loads(open('./intents.json', encoding = "utf8").read())
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
# Dự đoán lớp
def predict_classes(sentence, model):
    p = bag_element(sentence, words)
    respond = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(respond) if r>ERROR_THRESHOLD]
    # print(result)
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return result_list

# Phản hồi ngẫu nhiên từ chủ đề 
def getRespond (ints, intent_json):
    tag = ints[0]['intent']
    list_intents = intent_json['intents']
    for i in list_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
def bot_respond(text_input):
    ints = predict_classes(text_input, model)
    respond = getRespond(ints, intents)
    return respond

# ChatBot
while True:
    user = input("User: ")
    res = bot_respond(user)
    print("BOT: {}".format(res))