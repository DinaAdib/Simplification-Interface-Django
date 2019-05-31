import pyphen
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
from nltk.stem.porter import *
import os  # importing os to set environment variable
from wordfreq import zipf_frequency
import sys
from sklearn.preprocessing import maxabs_scale
import pickle


DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/files/'
######## General Functions ############
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

