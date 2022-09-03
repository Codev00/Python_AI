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