import nltk
# nltk.download('punkt')
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD 
import random

# Lưu trữ từ
words = []
# Lưu trữ chủ đề
classes = []
# Kho dữ liệu
documents = []
ignore_words = ['?', '!']
data_file = open('./intents.json', encoding = "utf8").read()
intents = json.loads(data_file)

# Chia văn bản thành các từ (Tokenizing)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Mã hoá từng từ 
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Thêm tài liệu vào kho dữ liệu
        documents.append((w, intent['tag']))
        # Thêm vào danh sách lớp cho máy học
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Bổ sung, xoá từ trùng lặp và sắp xếp từ
words = [w.lower() for w in words if w not in ignore_words]
words = sorted(list(set(words)))
 
# Sắp xếp lớp
classes = sorted(list(set(classes)))

# Tạo tệp để lưu văn bản chữ => số
pickle.dump(words,open("words.pkl", "wb"))
pickle.dump(classes,open("classes.pkl", "wb"))

# Tạo dữ liệu đào tạo
training = []

# Tạo mảng trống cho đầu ra
output_empty = [0] * len(classes)

# Huấn luyện, bag.array chứa từ cho mỗi câu
for doc in documents:
    bag = []
    # Danh sách các từ được mã hoá (documents chứa các từ vựng trong phần pattern)
    pattern_words = doc[0]
    # Biểu thị các từ có liên quan
    pattern_words = [word.lower() for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # Các vị trí ko xét = 0
    output_row = list(output_empty)
    # Đánh dấu vị trí hiện tại = 1
    output_row[classes.index(doc[1])] = 1
    # Tạo tập training với bag chứa các mẫu từ đã chuyển số đựng trong bag và đầu ra là câu trả lời tương ứng
    training.append([bag, output_row])


# Trộn các training và chuyển thành mảng (np.array)
random.shuffle(training)
training = np.array(training)
# print(training)
# Tạo danh sách kiểm tra x-mẫu, y-ý định
train_x = list(training[:, 0])
# print(train_x)
train_y = list(training[:, 1])
# print(train_y)

# Mô hình train
# Tạo neurons
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5)) # Dropout tránh over-fitting
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax")) # softmax hàm tính trọng số cho dữ liệu

# Dịch mô hình
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # decay để điều chỉnh tốc độ học tập(giảm tỷ lệ) tăng dộ chính xác cho dữ liệu
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit và lưu model
tots = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=20, verbose=1)
model.save('train_model.h5', tots)
print("Hoàn thành training model !!")