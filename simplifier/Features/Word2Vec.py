import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from simplifier.functions import *


class Word2Vec:
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            DIRECTORY + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.vocab = self.word2vec.vocab
        self.vec_size = self.word2vec.vector_size

    def get_vector(self, phrase):
        words = [word.lower() for word in phrase if word in self.vocab]
        # print(words)
        if len(words) > 0:
            embeddings = np.array([self.word2vec.word_vec(word) for word in words])
        else:
            embeddings = [[0] * self.vec_size]
        return np.average(embeddings, axis=0).tolist()

    def get_cosine_similarity(self, vector1, vector2):
        return cosine_similarity(vector1, vector2)[0][0]

# myword2vec = Word2Vec()
# print("loaded word2vec")
# np.save("word2vec", myword2vec)