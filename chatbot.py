import nltk
import pickle
import numpy as np 

from keras.models import load_model

import json
import random
model = load_model('train_model.h5')
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
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    result_list = []
    for r in results:
        result_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(result_list)
    return result_list

# Phản hồi ngẫu nhiên từ class xác định
def getRespond (ints, intent_json):
    tag = ints[0]['intent']
    list_intents = intent_json['intents']
    for i in list_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    print(result)
    return result
def bot_respond(text_input):
    ints = predict_classes(text_input, model)
    respond = getRespond(ints, intents)
    return respond

# ChatBot
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
            res = bot_respond(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

def send_enter(event):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
            res = bot_respond(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

noah = Tk()
noah.title("NOAH")
noah.geometry("400x500")
noah.resizable(width=FALSE, height=FALSE)

ChatLog = Text(noah, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(noah, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(noah, font=("Verdana",12,'bold'), text="Send", width="8", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
EntryBox = Text(noah, bd=0, bg="white",width="29", height="5", font="Arial")
noah.bind('<Return>', send_enter)
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=406, width=370)
EntryBox.place(x=6, y=431, height=50, width=290)
SendButton.place(x=291, y=431, height=50)

noah.mainloop()