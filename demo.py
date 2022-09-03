
# import spacy
# import random


# bot_template = "BOT : {0}"
# user_template = "USER : {0}"

# #Brain
# responses ={
#    "Tên bạn là gì?": [
#       "Tôi là Echobot",
#       "Cứ gọi tôi là EchoBot",
#       "Tôi á, EchoBot"
#     ],
#     "default": "Tôi vẫn chưa được học T_T"
# }
# # Định nghĩa một hàm để phản hồi tin nhắn của người dùng
# def respond(message):
#     # Nối tin nhắn của người dùng và tin nhắn phản hồi
#     if message in responses:
#         return random.choice(responses[message])
#     else:
#         return responses["default"]


# def send_message(message):
#     # In user_message sử dụng mẫu user_template
#     print(user_template.format(message))
#     # Lưu phản hồi của bot
#     response = respond(message)
#     # In phản hồi của bot sử dụng mẫu bot_template
#     print(bot_template.format(response))

# # Kiểm tra hàm vừa xây dựng
# send_message("Tên bạn là gì nè?")

# nlp = spacy.load("en_core_web_sm")
# print(nlp.vocab.vectors_length)

# import pandas as pd
# df = pd.read_csv("./converse.csv")
# print(df.head(180))
import numpy as np 
bag = np.zeros((2,2), dtype='int')
print(bag)